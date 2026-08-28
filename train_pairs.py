# Imports
import csv

import cv2
import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision import transforms

from CombinedLoss import CombinedLoss, FocalLoss, BinaryDiceLoss
from Models.DeepViTAS import DeepViTAS

# Other
import os
import numpy as np
import random
from tqdm import tqdm as tqdm

import time
import warnings
import psutil
import gc

from Utils.ChangeDetectionDataset import ChangeDetectionDataset, RandomRot, RandomFlip
from Utils.Evaluation import apply_metrics_altamira



PATH_CLUSTER = ''
PATH_DIR = 'ALTAMIRA_tests/'
PATH_TO_DATASET = PATH_CLUSTER + 'Datasets/Dataset_Altamira/'

SEED = 407
BATCH_SIZE = 24
PATCH_SIDE = 96
N_EPOCHS = 100
EMB_DIM = 512

# ===============================
# Settings seed
# ===============================
os.environ["PYTHONHASHSEED"] = str(SEED)

torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
np.random.seed(SEED)
random.seed(SEED)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.use_deterministic_algorithms(True, warn_only=True)


# ===============================
# Workers
# ===============================
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# Global deterministic generator
g = torch.Generator()
g.manual_seed(SEED)

TYPE = 0
NORMALISE_IMGS = True
TRAIN_STRIDE = int(PATCH_SIDE / 2) - 1
LOAD_TRAINED = False
DATA_AUG = True

FP_MODIFIER = 10
L = 1024
N = 2

print('DEFINITIONS OK')


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def train(n_epochs=N_EPOCHS, path_save="", name="", loss="combine"):
    # Hiperparâmetros

    if loss == "focal":
        criterion = FocalLoss(gamma=4.41)
    elif loss == "dice":
        criterion = BinaryDiceLoss(smooth=1.66)
    else:
        criterion = CombinedLoss(alpha=0.1, beta=0.9, smooth=1.66, gamma=4.41)

    # Variáveis de controle
    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0

    optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # CSV: path and head
    csv_path = os.path.join(path_save, f"{name}_training_log.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    start_time = time.time()
    for epoch_index in tqdm(range(n_epochs)):
        net.train()

        print('Epoch: ' + str(epoch_index + 1) + ' of ' + str(N_EPOCHS))

        running_loss = 0

        #         for batch_index, batch in enumerate(tqdm(data_loader)):

        for batch in train_loader:
            I1 = Variable(batch['I1'].float())
            # I2 = Variable(batch['I2'].float())
            label = torch.squeeze(Variable(batch['label']))
            I1 = I1.to(device)
            label = label.to(device)
            #print(I1.shape)
            #print(label.shape)

            optimizer.zero_grad()
            output = net(I1)
            loss = criterion(output, label.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        net.eval()
        val_loss = 0.0
        with torch.no_grad():
            for batch in val_loader:
                I1 = Variable(batch['I1'].float())
                label = torch.squeeze(Variable(batch['label']))
                I1 = I1.to(device)
                label = label.to(device)

                output = net(I1)
                loss = criterion(output, label.long())
                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        val_loss /= len(val_loader)  # Média da perda de validação
        # Atualizar o scheduler
        scheduler.step(val_loss)

        # Salvar as perdas no CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch_index + 1, avg_train_loss, val_loss])

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            epochs_no_improve = 0
            save_str = path_save + '/' + name
            torch.save(net.state_dict(), save_str + '.pth')
            torch.save(net, save_str + "_model.pth")
            print("✅ Model improved and saved!")


        else:
            epochs_no_improve += 1
            print(f"⏳ There has been no improvement for {epochs_no_improve} epoch(s).")

            if epochs_no_improve >= patience:
                print(f"⛔ Early stopping was activated after {epoch_index} epochs with no improvement.")
                break

    # Tempo total
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"⏱️Total training time: {int(mins)}m {int(secs)}s")

    # Salvar tempo total no .csv
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Total training time", f"{int(mins)}m {int(secs)}s"])


def segment_and_time_combine(path_model, test_dset, path_save_segmented):
    # Limpar cache
    torch.cuda.empty_cache()
    torch.cuda.synchronize()

    # Carregar modelo
    state_dict = torch.load(path_model, map_location=device)
    del state_dict["vit.patch_embed.pos_embed"]
    net.load_state_dict(state_dict, strict=False)
    net.to(device)
    net.eval()

    times = []

    # Aquecimento
    with torch.no_grad():
        for name in tqdm(test_dset.names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                I1, cm = test_dset.get_img(name)
                I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)
                _ = net(I1)
                torch.cuda.synchronize()
                break

    # Loop principal
    with torch.no_grad():
        for name in tqdm(test_dset.names):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")

                I1, cm = test_dset.get_img(name)
                I1 = Variable(torch.unsqueeze(I1, 0).float()).to(device)

                torch.cuda.synchronize()
                start_time = time.time()

                output = net(I1)

                torch.cuda.synchronize()
                end_time = time.time()

                # Tempo
                execution_time = end_time - start_time
                times.append(execution_time)

            # Processar e salvar
            _, predicted = torch.max(output.data, 1)
            output_np = 255 * np.squeeze(predicted.cpu().numpy()).astype(np.uint8)

            original_name = name + '.png'

            save_path = os.path.join(path_save_segmented, original_name)
            cv2.imwrite(save_path, output_np)

    print(f"✅ Segmentation complete! Images saved to: {path_save_segmented}")
    print(f"⏱️Average execution time: {np.mean(times):.6f} seconds")
    return times




def verify_memory(process):
    # Get the memory usage in bytes
    memory_info = process.memory_info()
    memory_used = memory_info.rss  # Resident memory usage in bytes

    # Convert to megabytes
    memory_used_mb = memory_used / (1024 ** 2)

    print(f"Memory used: {memory_used_mb:.2f} MB")

import os
import psutil
import torch
import gc





if __name__ == "__main__":
    cache_dir = PATH_CLUSTER + 'newcache'
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)  # Write permissions

    os.environ['TORCH_HOME'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)

    # Global Variables' Definitions
    # Get the ID of the current process
    # pid = os.getpid()
    # # Creates a Process object for the current process
    # process = psutil.Process(pid)
    #
    # verify_memory(process)

    if DATA_AUG:
        # ===============================
        # 🚀 5. Composition of Transformations
        # ===============================
        data_transform = transforms.Compose([
            RandomFlip(generator=g),
            RandomRot(generator=g)
        ])
    else:
        data_transform = None

    train_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=0, stride=TRAIN_STRIDE, transform=data_transform,
                                           patch_side=PATCH_SIDE)
    val_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=1, stride=TRAIN_STRIDE, patch_side=PATCH_SIDE)
    test_dataset = ChangeDetectionDataset(PATH_TO_DATASET, train=2, stride=TRAIN_STRIDE)

    # ----------------------------
    # Set Seed and Dataloaders
    # ----------------------------
    print("\n\n Criando Dataloaders \n")

    # ===============================
    # 📥 6. DataLoader
    # ===============================
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        generator=g,
        pin_memory=True
    )

    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0,
                            pin_memory=True, generator=g)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0,
                             pin_memory=True)

    gt_path = "Datasets/Dataset_Altamira/"

    print('DATASETS OK')


    # -------------------------------------------------------------------------------------

    print('\n\n' + PATH_DIR + 'DeepViTAS\n\n')
    os.makedirs(PATH_DIR + "DeepViTAS", exist_ok=True)

    net = DeepViTAS(in_channels=3, out_channels=2, embed_dim=EMB_DIM, mode='bicubic')
    net_name = 'deepvitas128'
    net.to(device)
    criterion = CombinedLoss(alpha=0.1, beta=0.9, smooth=1.66, gamma=4.41)


    train(path_save=PATH_DIR + "DeepViTAS", name=net_name)

    # -------------------------------------------------------------------------------------




    print("\n\nStarting the Test")
    # ----------------------------------------------------------------------------
    print('\n\n DeepViTAS \n\n')
    path_save = PATH_DIR + "DeepViTAS/Results"
    os.makedirs(path_save, exist_ok=True)

    net = DeepViTAS(in_channels=3, out_channels=2, embed_dim=EMB_DIM, mode='bicubic').to(device)
    dummy = torch.randn(1, 3, 640, 640).to(device)
    _ = net(dummy)

    model_path = PATH_DIR + 'DeepViTAS/' + 'deepvitas128.pth'

    segment_and_time_combine(model_path, test_dataset, path_save)
    apply_metrics_altamira(path_save + "/Altamira_2020_", gt_path)

    # ----------------------------------------------------------------------------

