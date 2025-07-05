import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import os
import sys

# Impor library yang dibutuhkan untuk laporan dan plot
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Menambahkan parent directory ke path agar bisa mengimpor modul
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Impor dari file proyek Anda
from crnn_model import CRNN_Attention 
from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from train import collate_fn

def generate_final_report():
    print("--- Memulai Ujian Akhir pada Data Test Kaggle (Data Tak Tersentuh) ---")
    
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    device = torch.device(config["device"])
    charset = config["dataset"]["charset"]

    le = LabelEncoder(charset)
    # PENTING: Arahkan ke file label test dari Kaggle
    test_ds = OCRDataset(
        'dataset/kaggle_icdar_val_labels.txt',
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le,
        is_train=False
    )
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=0, collate_fn=collate_fn)

    num_classes = len(charset) + 1
    model = CRNN_Attention(
        config["dataset"]["img_height"], 1, num_classes
    ).to(device)
    # Menggunakan weights_only=True adalah praktik yang lebih aman
    model.load_state_dict(torch.load(config["training"]["model_save_path"], map_location=device, weights_only=True))
    model.eval()

    all_preds_text = []
    all_true_text = []
    all_preds_chars = []
    all_true_chars = []

    with torch.no_grad():
        for images, labels, lengths in tqdm(test_loader, desc="Menjalankan Ujian Akhir"):
            logits = model(images.to(device))
            preds_tokens = logits.argmax(2).squeeze(0).cpu().numpy().tolist()
            pred_text = le.decode(preds_tokens)
            true_text = le.decode(labels[0, :lengths[0]].cpu().numpy().tolist())
            
            all_preds_text.append(pred_text)
            all_true_text.append(true_text)
            
            # Samakan panjang untuk perbandingan karakter
            len_min = min(len(pred_text), len(true_text))
            all_preds_chars.extend(list(pred_text)[:len_min])
            all_true_chars.extend(list(true_text)[:len_min])

    print("\n--- LAPORAN KINERJA FINAL ---")
    word_accuracy = accuracy_score(all_true_text, all_preds_text)
    print(f"Akurasi Final Per Kata (Word Accuracy) di Test Set: {word_accuracy:.4f}")
    print("-" * 30)

    # Gunakan daftar karakter yang sudah disamakan panjangnya
    report = classification_report(
        all_true_chars, 
        all_preds_chars,
        labels=sorted(list(charset)), 
        zero_division=0
    )
    print("Laporan Kinerja per Karakter di Test Set:")
    print(report)

    # =============================================================
    # --- TAMBAHAN: Membuat dan Menyimpan Confusion Matrix ---
    print("\nMembuat Confusion Matrix...")
    all_characters = sorted(list(charset))
    cm = confusion_matrix(all_true_chars, all_preds_chars, labels=all_characters)
    
    # Membuat plot menggunakan seaborn untuk visualisasi yang lebih baik
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=all_characters, yticklabels=all_characters)
    plt.title('Final Confusion Matrix - Kinerja per Karakter di Test Set', fontsize=20)
    plt.xlabel('Karakter Prediksi', fontsize=16)
    plt.ylabel('Karakter Sebenarnya', fontsize=16)
    
    # Simpan plot ke file
    save_path = "final_confusion_matrix.png"
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix berhasil disimpan di: {save_path}")
    # =============================================================

if __name__ == "__main__":
    generate_final_report()