import torch
from torch.utils.data import DataLoader
import yaml
from tqdm import tqdm
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from Levenshtein import distance as levenshtein_distance

# Impor dari file proyek Anda
from crnn_model import CRNN_Attention
from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from train import collate_fn # Pastikan collate_fn sudah di-update

def evaluate():
    # 1. Muat Konfigurasi
    with open("config.yaml") as f:
        config = yaml.safe_load(f)
    device = torch.device(config["device"])
    charset = config["dataset"]["charset"]

    # 2. Siapkan Dataset Validasi
    le = LabelEncoder(charset)
    val_ds = OCRDataset(
        config["dataset"]["val_labels"],
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le,
        is_train=False
    )
    val_loader = DataLoader(
        val_ds, batch_size=1, shuffle=False,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn
    )

    # 3. Muat Model Terbaik
    num_classes = len(charset) + 1
    model = CRNN_Attention(
        config["dataset"]["img_height"], 1, num_classes
    ).to(device)
    model.load_state_dict(torch.load(config["training"]["model_save_path"], map_location=device, weights_only=True))
    model.eval()

    print("Model berhasil dimuat. Memulai evaluasi...")

    # 4. Lakukan Prediksi dan Kumpulkan Hasil
    all_preds_text = []
    all_true_text = []
    all_preds_chars = []
    all_true_chars = []
    total_edit_distance = 0
    total_char_length = 0
    error_samples = []

    with torch.no_grad():
        # Sekarang loopnya juga mengeluarkan image_paths
        for images, labels, lengths, image_paths in tqdm(val_loader, desc="Mengevaluasi Model"):
            images = images.to(device)
            logits = model(images)
            log_probs = logits.log_softmax(2)
            preds_tokens = log_probs.argmax(2).squeeze(0).cpu().numpy().tolist()
            pred_text = le.decode(preds_tokens)
            true_text = le.decode(labels[0, :lengths[0]].cpu().numpy().tolist())

            all_preds_text.append(pred_text)
            all_true_text.append(true_text)
            
            len_min = min(len(pred_text), len(true_text))
            all_preds_chars.extend(list(pred_text)[:len_min])
            all_true_chars.extend(list(true_text)[:len_min])
            
            edit_distance = levenshtein_distance(pred_text, true_text)
            total_edit_distance += edit_distance
            total_char_length += len(true_text)
            
            # --- PERUBAHAN UTAMA DI SINI ---
            # Simpan juga path gambarnya jika terjadi error
            if edit_distance > 0:
                # image_paths adalah tuple, kita ambil elemen pertamanya
                error_samples.append((edit_distance, true_text, pred_text, image_paths[0]))
            # --------------------------------

    # 5. Hitung dan Tampilkan Metrik
    print("\n--- Laporan Evaluasi Model Komprehensif ---")
    word_accuracy = accuracy_score(all_true_text, all_preds_text)
    print(f"✅ Akurasi Per Kata (Word Accuracy): {word_accuracy:.4f} ({word_accuracy:.2%})")
    cer = total_edit_distance / total_char_length
    print(f"✅ Character Error Rate (CER): {cer:.4f} ({cer:.2%})")
    print("-" * 45)

    print("📊 Laporan Detail Kinerja per Karakter:")
    all_characters = sorted(list(charset))
    report = classification_report(all_true_chars, all_preds_chars, labels=all_characters, zero_division=0)
    print(report)
    print("-" * 45)

    # --- PERUBAHAN UTAMA DI SINI ---
    print("🔬 Analisis Sampel Prediksi Terburuk (Top 15):")
    error_samples.sort(key=lambda x: x[0], reverse=True)
    # Membuat header tabel yang lebih panjang
    print("{:<20} | {:<20} | {:<10} | {:<40}".format("Teks Sebenarnya", "Prediksi Model", "Jarak Error", "Path Gambar"))
    print("-" * 95)
    # Menampilkan path gambar di dalam loop
    for distance, true, pred, path in error_samples[:15]:
        print("{:<20} | {:<20} | {:<10} | {:<40}".format(true, pred, distance, path))
    print("-" * 45)
    # ------------------------------------

    # 6. Buat dan Simpan Confusion Matrix
    print("\n📊 Membuat Confusion Matrix...")
    # ... (kode confusion matrix tetap sama) ...
    cm = confusion_matrix(all_true_chars, all_preds_chars, labels=all_characters)
    plt.figure(figsize=(20, 16))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=all_characters, yticklabels=all_characters)
    plt.title('Confusion Matrix Kinerja per Karakter', fontsize=20)
    plt.xlabel('Karakter Prediksi', fontsize=16)
    plt.ylabel('Karakter Sebenarnya', fontsize=16)
    save_path = "confusion_matrix.png"
    plt.savefig(save_path)
    print(f"✅ Confusion Matrix berhasil disimpan di: {save_path}")

if __name__ == "__main__":
    evaluate()