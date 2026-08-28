#Load
# -----------------------
# PyTorch Libraries
# -----------------------
import gc
import random

import psutil
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader


# -----------------------
# Scientific and data-handling libraries
# -----------------------
import numpy as np
from tqdm import tqdm

# -----------------------
# Reading Georeferenced Images and Image Processing
# -----------------------
import rioxarray as rxr
import cv2
import time
import csv
from Utils.Evaluation import apply_metrics_amazon
# -----------------------
# File and System Management
# -----------------------
import os
import glob


# -----------------------
# Models
# -----------------------
from Models.DeepViTAS import DeepViTAS

# -----------------------
# Loss
# -----------------------
from CombinedLoss import CombinedLoss


#--------------------------------------
# Load Dataset
#--------------------------------------
from Utils.SatelliteDataset import SatelliteDataset


def load_images(path, channels=4):
    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))
    images = []
    names = []

    for file in file_list:
        img = np.array(rxr.open_rasterio(file), dtype=np.float32)
        img = (img - np.min(img)) / (np.max(img) - np.min(img) + 1e-8)  # Normalização [0,1
        images.append(img)
        names.append(os.path.basename(file))

    images = np.array(images)
    return images, names


def load_masks(path):
    file_list = sorted(glob.glob(os.path.join(path, "*.tif")))

    # Load and invert the masks
    masks = [np.array(rxr.open_rasterio(file), dtype=np.float32) for file in file_list]
    masks = np.array(masks)
    names = [file for file in file_list]
    names = np.array(names)

    # Verify normalization (0-1) or (0-255)
    if masks.max() > 1:
        masks = 255 - masks  # Invert (0↔255)
    else:
        masks = 1 - masks  # Invert (0↔1)

    return masks, file_list





#--------------------------------------------------
# Traning Functions
#--------------------------------------------------


def train_combine(model, train_loader, val_loader, num_epochs, name, path_save):
    # Hiperparâmetros
    learning_rate = 0.0001
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_fn = CombinedLoss(alpha=0.1, beta=0.9, smooth=1.66, gamma=4.41)

    # Variáveis de controle
    best_val_loss = float("inf")
    patience = 10
    epochs_no_improve = 0


    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # CSV: caminho e cabeçalho
    csv_path = os.path.join(path_save, f"{name}_training_log.csv")
    with open(csv_path, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Train Loss", "Val Loss"])

    # Medir tempo de treinamento
    start_time = time.time()

    # Loop de treinamento
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            images, masks = images.to(device), masks.to(device)
            masks = masks.float()
            masks = torch.squeeze(masks, dim=1)

            optimizer.zero_grad()
            outputs = model(images)
            loss = loss_fn(outputs, masks.long())

            if torch.isnan(loss) or torch.isinf(loss):
                print("⚠️ Invalid loss - ignoring batch")
                continue

            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        # Validação
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                masks = masks.float() #/ (masks.max() + 1e-8)
                masks = torch.squeeze(masks, dim=1)

                outputs = model(images)
                loss = loss_fn(outputs, masks.long())

                if torch.isnan(loss) or torch.isinf(loss):
                    print("⚠️ Val Loss contains NaN or Inf!")
                    continue

                val_loss += loss.item()

        avg_train_loss = running_loss / len(train_loader)
        avg_val_loss = val_loss / len(val_loader)
        scheduler.step(avg_val_loss)

        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

        # Salvar as perdas no CSV
        with open(csv_path, mode='a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([epoch + 1, avg_train_loss, avg_val_loss])

        # Early stopping
        if avg_val_loss < best_val_loss - 1e-4:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0

            torch.save(model.state_dict(), os.path.join(path_save, name + ".pth"))
            torch.save(model, os.path.join(path_save, name + "model.pth"))
            print("✅ Model improved and saved!")
        else:
            epochs_no_improve += 1
            print(f"⏳ No improvement for {epochs_no_improve} epoch(s)")

            if epochs_no_improve >= patience:
                print(f"⛔ Early stopping enabled after {epoch} epochs with no improvement.")
                break

    # Tempo total
    end_time = time.time()
    elapsed = end_time - start_time
    mins, secs = divmod(elapsed, 60)
    print(f"⏱️ Total training time: {int(mins)}m {int(secs)}s")

    # Salvar tempo no .txt
    with open(os.path.join(path_save, "training_time.txt"), "w") as f:
        f.write(f"Total training time: {int(mins)}m {int(secs)}s\n")

    # Salvar tempo total no .csv
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([])
        writer.writerow(["Total training time", f"{int(mins)}m {int(secs)}s"])


#------------------------------------------------------
# Funções de Teste
#-------------------------------------------------------




def segment_and_time_combine(path_model, test_loader, path_save_segmented):
    # Limpar cache
    #torch.cuda.empty_cache()
    #torch.cuda.synchronize()

    # Carregar modelo
    model.load_state_dict(torch.load(path_model, map_location=device))
    model.to(device)
    model.eval()

    times = []

    # Aquecimento
    with torch.no_grad():
        for image, _, _ in test_loader:
            image = image.to(device)
            _ = model(image)
            #torch.cuda.synchronize()
            break

    # Loop principal
    with torch.no_grad():
        for image, _, filename in tqdm(test_loader, desc="Segmentando imagens"):
            image = image.to(device)

            #torch.cuda.synchronize()
            start_time = time.time()

            output = model(image)

            #torch.cuda.synchronize()
            end_time = time.time()

            # Tempo
            execution_time = end_time - start_time
            times.append(execution_time)

            # Processar e salvar
            _, predicted = torch.max(output.data, 1)
            output_np = 255 * np.squeeze(predicted.cpu().numpy()).astype(np.uint8)

            original_name = filename[0].replace('.tif', '.png')
            save_path = os.path.join(path_save_segmented, original_name)
            cv2.imwrite(save_path, output_np)

    print(f"✅ Segmentation complete! Images saved to: {path_save_segmented}")
    print(f"⏱ Average execution time: {np.mean(times):.6f} seconds")
    return times




if __name__ == "__main__":
    PATH_CLUSTER = ""
    PATH_ROOT = os.path.join(PATH_CLUSTER, "AMAZON_tests/")
    PATH_DATASET = os.path.join(PATH_CLUSTER, "Datasets/AMAZON/")
    PATH_GT = os.path.join(PATH_DATASET, "Test/mask_png/")

    PATH_DIR = os.path.join(PATH_ROOT, "100EP_SEED723_16B")

    NUM_EPOCHS = 100
    SEED = 723
    BATCH_SIZE = 16
    EMB_DIM = 128


    cache_dir = PATH_CLUSTER + 'newcache'
    os.makedirs(cache_dir, exist_ok=True)
    os.chmod(cache_dir, 0o777)

    os.environ['TORCH_HOME'] = cache_dir
    os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"\n Device: {device}\n")

    #------------------------------
    # Load Datasets
    #------------------------------
    print("\n Loading Data \n")
    training_images2, _ = load_images(os.path.join(PATH_DATASET, "Training/images"))
    training_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Training/masks"))

    validation_images2, _ = load_images(os.path.join(PATH_DATASET, "Validation/images"))
    validation_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Validation/masks"))

    test_images2, names = load_images(os.path.join(PATH_DATASET, "Test/images"))
    test_masks2, _ = load_masks(os.path.join(PATH_DATASET, "Test/masks"))

    # Verify formats
    print(f"📌 Training - Images: {training_images2.shape}, Masks: {training_masks2.shape}")
    print(f"📌 Test - Images: {test_images2.shape}, Masks: {test_masks2.shape}")
    print(f"📌 Validation - Images: {validation_images2.shape}, Masks: {validation_masks2.shape}")


    #----------------------------
    # Create Datasets
    #----------------------------
    print("\n\n Create Datasets \n")
    train_dataset = SatelliteDataset(training_images2, training_masks2)
    val_dataset = SatelliteDataset(validation_images2, validation_masks2)
    test_dataset = SatelliteDataset(test_images2, test_masks2, filenames=names)

    #----------------------------
    # Set Seed and Dataloaders
    #----------------------------
    print("\n\n Create Dataloaders \n")
    g = torch.Generator()
    g.manual_seed(SEED)
    g.manual_seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    np.random.seed(SEED)
    random.seed(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    #For GPU
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True, generator=g)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)

    #For CPU
    #train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=True, generator=g)
    #val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, num_workers=0, shuffle=False)
    #test_loader = DataLoader(test_dataset, batch_size=1, num_workers=0, shuffle=False)


    #-----------------------------------
    # Trainig
    #-----------------------------------
    print("\n\nDeepViTAS\n\n")
    path_save = os.path.join(PATH_DIR, "DeepViTAS_AMAZON/")
    os.makedirs(path_save, exist_ok=True)

    # Create an instance of the model
    model = DeepViTAS(in_channels=4, out_channels=2, mode='bicubic', embed_dim=EMB_DIM).to(device)
    criterion = CombinedLoss(alpha=0.1, beta=0.9, smooth=1.66, gamma=4.41)
    train_combine(model, train_loader, val_loader, NUM_EPOCHS, "deepvitas", path_save)

    #--------------------------------------------------
    print("End training session\n")



    #-----------------------------------------
    print("\nStart tests\n\n")
    # -----------------------------------------
    print("\n\nDeepViTAS\n")
    path_model = os.path.join(PATH_DIR, "DeepViTAS_AMAZON/deepvitas.pth")
    path_save_segmented = os.path.join(PATH_DIR, "DeepViTAS_AMAZON/Results/")
    os.makedirs(path_save_segmented, exist_ok=True)


    model = DeepViTAS(in_channels=4, out_channels=2, mode='bicubic', embed_dim=EMB_DIM).to(device)
    dummy = torch.randn(1, 4, 512, 512).to(device)
    _ = model(dummy)  # creates self.vit
    segment_and_time_combine(path_model, test_loader, path_save_segmented)
    apply_metrics_amazon(path_save_segmented, PATH_GT)
















