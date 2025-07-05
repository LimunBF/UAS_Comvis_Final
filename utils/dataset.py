import os
import cv2
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2

class OCRDataset(Dataset):
    def __init__(self, label_path, img_height, img_width, label_encoder, is_train=True):
        self.label_encoder = label_encoder
        self.is_train = is_train
        self.label_path = label_path # <-- TAMBAHKAN BARIS INI
        self.data = []

        # Asumsikan path di dalam file .txt sudah benar relatif dari root folder project Anda
        # Contoh isi baris di train_labels.txt: dataset/train_images/1.png  hello
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f.readlines():
                parts = line.strip().split('\t')
                if len(parts) == 2:
                    img_path, label_text = parts
                    self.data.append((img_path, label_text))

        # Pipeline augmentasi untuk TRAINING (disederhanakan untuk menghindari warning)
        if self.is_train:
            self.transform = A.Compose([
                A.Rotate(limit=5, p=0.5, border_mode=cv2.BORDER_CONSTANT),
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
                A.GaussianBlur(blur_limit=(3, 5), p=0.3),
                A.GaussNoise(p=0.3),
                A.Resize(height=img_height, width=img_width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
        # Transformasi sederhana untuk VALIDASI
        else:
            self.transform = A.Compose([
                A.Resize(height=img_height, width=img_width, interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, label = self.data[idx]

        # Baca gambar sebagai Grayscale
        image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        # Pemeriksaan jika gambar gagal dimuat (ini akan mengatasi TypeError)
        if image is None:
            raise FileNotFoundError(
                f"FATAL: Gambar tidak ditemukan atau rusak di path: {image_path}. "
                f"Pastikan path di dalam '{self.label_path}' sudah benar."
            )

        # Tambahkan channel dimension: (H, W) -> (H, W, 1)
        image = image[:, :, None]

        # Terapkan transformasi
        transformed = self.transform(image=image)
        image_tensor = transformed['image']

        # Encode label
        label_encoded, length = self.label_encoder.encode(label.lower())

        # di utils/dataset.py, di akhir fungsi __getitem__
        return image_tensor, torch.tensor(label_encoded, dtype=torch.long), length, image_path