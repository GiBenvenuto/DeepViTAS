from torch.utils.data import Dataset, DataLoader
import torch


class SatelliteDataset(Dataset):
    def __init__(self, images, masks, filenames=None, transform=None):
        self.images = images
        self.masks = masks
        self.filenames = filenames if filenames is not None else [None] * len(images)  # Garante compatibilidade
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image = self.images[idx]
        mask = self.masks[idx]
        filename = self.filenames[idx]  # Pode ser None se não for fornecido

        # Converter para tensor (caso ainda não esteja)
        image = torch.tensor(image, dtype=torch.float32)
        mask = torch.tensor(mask, dtype=torch.long)  # Máscaras geralmente são rótulos inteiros

        # Aplicar transformações (caso existam)
        if self.transform:
            image = self.transform(image)

        if filename is not None:
            return image, mask, filename
        else:
            return image, mask
