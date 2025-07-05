import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import yaml
from crnn_model import CRNN_Attention
from utils.dataset import OCRDataset
from utils.label_encoder import LabelEncoder
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from Levenshtein import distance  # untuk hitung CER
# Impor loss baru kita
from utils.loss import CTCLossWithLabelSmoothing

def collate_fn(batch):
    # Sekarang kita unpack 4 item, termasuk image_paths
    images, labels, lengths, image_paths = zip(*batch)
    images = torch.stack(images)
    labels = pad_sequence(labels, batch_first=True, padding_value=0)
    
    # Kita kembalikan juga image_paths
    return images, labels, torch.tensor(lengths, dtype=torch.long), image_paths

def ctc_decode(preds, blank=0):
    decoded = []
    prev = blank
    for p in preds:
        if p != blank and p != prev:
            decoded.append(p)
        prev = p
    return decoded

def train():
    # Load konfigurasi
    with open("config.yaml") as f:
        config = yaml.safe_load(f)

    # Set seed & device
    torch.manual_seed(config["seed"])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(config["seed"])
    device = torch.device(config["device"])

    # Dataset & DataLoader
    le = LabelEncoder(config["dataset"]["charset"])
    train_ds = OCRDataset(
        config["dataset"]["train_labels"],
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le,
        is_train=True
    )
    val_ds = OCRDataset(
        config["dataset"]["val_labels"],
        config["dataset"]["img_height"],
        config["dataset"]["img_width"],
        le,
        is_train=False
    )
    train_loader = DataLoader(
        train_ds,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=1,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_fn
    )

    # Model & loss
    num_classes = len(config["dataset"]["charset"]) + 1
    model = CRNN_Attention(
        config["dataset"]["img_height"], 1, num_classes
    ).to(device)
    # Ganti criterion lama:
    # criterion = nn.CTCLoss(blank=0, zero_infinity=True)

    # Menjadi criterion baru dengan Label Smoothing:
    criterion = CTCLossWithLabelSmoothing(num_classes=num_classes, blank=0, smoothing=0.1)

    # --- PERUBAHAN 1: Menggunakan Optimizer AdamW ---
    optimizer = optim.AdamW(
        model.parameters(), 
        lr=config["training"]["learning_rate"],
        weight_decay=0.01 # Nilai weight decay yang umum untuk AdamW
    )
    # --- PERUBAHAN 2: Menggunakan Scheduler OneCycleLR ---
    # Logika Early Stopping lama (patience, min_delta, dll.) dihapus
    
    # Kita butuh total langkah training untuk mengonfigurasi scheduler ini
    total_steps = len(train_loader) * config["training"]["epochs"]
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=config["training"]["learning_rate"],
        total_steps=total_steps,
        pct_start=0.3, # 30% dari waktu digunakan untuk menaikkan LR
        anneal_strategy='cos' # Metode penurunan LR: cosine
    )
    
    # Kita tetap butuh variabel ini untuk menyimpan model terbaik
    best_val_loss = float("inf")

    for epoch in range(1, config["training"]["epochs"] + 1):
        model.train()
        total_loss = 0
        for images, labels, lengths in tqdm(train_loader, desc=f"Epoch {epoch} (Train)"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            log_probs = logits.log_softmax(2).permute(1, 0, 2)
            input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
            loss = criterion(log_probs, labels, input_lengths, lengths)
            loss.backward()
            optimizer.step()

            # --- PERUBAHAN 3: Step Scheduler Setiap Batch ---
            # OneCycleLR di-update setiap kali optimizer melangkah
            scheduler.step()
            # -----------------------------------------------------------
            
            total_loss += loss.item()
        avg_train_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch} Train Loss: {avg_train_loss:.4f}")

        # Validasi (Blok ini tidak ada perubahan)
        model.eval()
        val_loss = 0
        total_cer = 0
        with torch.no_grad():
            for images, labels, lengths in tqdm(val_loader, desc=f"Epoch {epoch} (Val)"):
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                log_probs = logits.log_softmax(2).permute(1, 0, 2)
                input_lengths = torch.full((log_probs.size(1),), log_probs.size(0), dtype=torch.long)
                loss = criterion(log_probs, labels, input_lengths, lengths)
                val_loss += loss.item()
                preds_raw = logits.argmax(2)[0].cpu().numpy().tolist()
                preds = ctc_decode(preds_raw, blank=0)
                pred_text = le.decode(preds)
                true_text = le.decode(labels[0, :lengths[0]].cpu().numpy().tolist())
                total_cer += distance(pred_text, true_text) / max(len(true_text), 1)
        avg_val_loss = val_loss / len(val_loader)
        avg_cer = total_cer / len(val_loader)
        print(f"Epoch {epoch} Val Loss: {avg_val_loss:.4f}, Val CER: {avg_cer:.4f}")

        # Contoh prediksi debug tiap 5 epoch
        if epoch % 5 == 0:
            print(f"Sample prediction: '{pred_text}'")
            print(f"Sample ground truth: '{true_text}'")

        # --- PERUBAHAN 4: Logika Penyimpanan Model (Tanpa Early Stopping) ---
        # Kita hanya menyimpan model jika validation loss-nya lebih baik dari sebelumnya.
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), config["training"]["model_save_path"])
            print(f"✅ Saved best model with validation loss: {best_val_loss:.4f}")
        # -----------------------------------------------------------

if __name__ == "__main__":
    train()