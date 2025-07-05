import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import cv2
import torch
import yaml
import os
import sys

# Menambahkan parent directory ke path agar bisa mengimpor modul dari folder lain
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from crnn_model import CRNN_Attention
from utils.label_encoder import LabelEncoder
import albumentations as A
from albumentations.pytorch import ToTensorV2
import easyocr

class OCR_GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Aplikasi Final Testing OCR")
        self.root.geometry("850x800") # Memberi ukuran awal pada window

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.label_encoder = None
        self.transform = None
        
        print("Memuat model deteksi EasyOCR... (Mungkin butuh beberapa saat saat pertama kali)")
        self.detector = easyocr.Reader(['en'], gpu=torch.cuda.is_available())
        print("Model deteksi EasyOCR berhasil dimuat.")

        self.load_config_and_model()

        # --- Tampilan GUI ---
        self.title_label = tk.Label(root, text="Aplikasi Final Testing OCR", font=("Helvetica", 16, "bold"))
        self.title_label.pack(pady=10)

        self.canvas = tk.Canvas(root, width=800, height=550, bg="lightgray")
        self.canvas.pack(padx=10, pady=5)

        # --- PERUBAHAN 1: Membuat Frame untuk Hasil Teks ---
        self.result_frame = tk.Frame(root, pady=10)
        self.result_frame.pack()
        
        self.result_header_label = tk.Label(self.result_frame, text="Hasil Pengenalan:", font=("Helvetica", 12, "bold"))
        self.result_header_label.pack()

        # Label ini akan kita update dengan hasil prediksi
        self.result_text_label = tk.Label(self.result_frame, text="- Unggah gambar untuk memulai -", font=("Helvetica", 12), wraplength=800)
        self.result_text_label.pack()
        # --------------------------------------------------

        self.btn_upload = tk.Button(root, text="Upload dan Analisis Gambar", font=("Helvetica", 12), command=self.upload_and_process)
        self.btn_upload.pack(pady=15)

    def load_config_and_model(self):
        print("Memuat konfigurasi dan model CRNN Anda...")
        try:
            config_path = os.path.join(os.path.dirname(__file__), '..', 'config.yaml')
            with open(config_path) as f:
                config = yaml.safe_load(f)
                model_path_rel = config["training"]["model_save_path"]

            self.label_encoder = LabelEncoder(config["dataset"]["charset"])
            self.transform = A.Compose([
                A.Resize(height=config["dataset"]["img_height"], width=config["dataset"]["img_width"], interpolation=cv2.INTER_LINEAR),
                A.Normalize(mean=[0.5], std=[0.5]),
                ToTensorV2(),
            ])
            num_classes = len(config["dataset"]["charset"]) + 1
            self.model = CRNN_Attention(
                config["dataset"]["img_height"], 1, num_classes
            ).to(self.device)
            model_path_abs = os.path.join(os.path.dirname(__file__), '..', model_path_rel)
            self.model.load_state_dict(torch.load(model_path_abs, map_location=self.device, weights_only=True))
            self.model.eval()
            print(f"Model CRNN Anda dari '{model_path_abs}' berhasil dimuat.")
        except FileNotFoundError as e:
            print(f"Error: Tidak dapat menemukan file! Pastikan path benar. Detail: {e}")
            self.root.destroy()

    def preprocess_cropped_image(self, cropped_img):
        if len(cropped_img.shape) == 3:
            cropped_img = cv2.cvtColor(cropped_img, cv2.COLOR_BGR2GRAY)
        cropped_img = cropped_img[:, :, None]
        transformed = self.transform(image=cropped_img)
        image_tensor = transformed['image'].unsqueeze(0).to(self.device)
        return image_tensor

    def upload_and_process(self):
        filepath = filedialog.askopenfilename()
        if not filepath: return
        
        # Bersihkan label hasil sebelumnya
        self.result_text_label.config(text="Menganalisis...")
        self.root.update_idletasks() # Paksa GUI untuk update
        
        image_cv = cv2.imread(filepath)
        detections = self.detector.readtext(image_cv)
        result_image = image_cv.copy()
        
        recognized_words = [] # Daftar untuk menampung kata yang berhasil dikenali

        for (bbox, _, _) in detections:
            (tl, tr, br, bl) = bbox
            tl = (int(tl[0]), int(tl[1]))
            br = (int(br[0]), int(br[1]))

            cropped_word = image_cv[tl[1]:br[1], tl[0]:br[0]]
            if cropped_word.size == 0: continue

            image_tensor = self.preprocess_cropped_image(cropped_word)
            with torch.no_grad():
                logits = self.model(image_tensor)
                preds_tokens = logits.argmax(2).squeeze(0).cpu().numpy().tolist()
                recognized_text = self.label_encoder.decode(preds_tokens)

            # Logika status Gagal/Berhasil
            if not recognized_text:
                display_text = "[Gagal]"
                text_color = (0, 0, 255)
            else:
                display_text = recognized_text
                text_color = (0, 255, 0)
                recognized_words.append(recognized_text) # Tambahkan ke daftar jika berhasil

            # Logika penempatan teks cerdas
            text_y_pos = tl[1] - 15
            if text_y_pos < 15: text_y_pos = br[1] + 15
            
            # Gambar kotak dan teks di atas gambar
            cv2.rectangle(result_image, tl, br, text_color, 2)
            cv2.putText(result_image, display_text, (tl[0], text_y_pos), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, text_color, 2)

        # --- PERUBAHAN 2: Update Label Hasil di GUI ---
        if not recognized_words:
            summary_text = "Tidak ada teks yang berhasil dikenali oleh model."
        else:
            summary_text = ", ".join(recognized_words)
        self.result_text_label.config(text=summary_text)
        # ---------------------------------------------
        
        self.display_image(result_image)

    def display_image(self, image_cv):
        image_rgb = cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(image_rgb)
        canvas_w, canvas_h = 800, 550
        pil_img.thumbnail((canvas_w, canvas_h))
        self.tk_img = ImageTk.PhotoImage(image=pil_img)
        self.canvas.delete("all")
        self.canvas.create_image(canvas_w/2, canvas_h/2, anchor='center', image=self.tk_img)

if __name__ == "__main__":
    main_root = tk.Tk()
    app = OCR_GUI(main_root)
    main_root.mainloop()