import os

def process_gt_file(data_dir, gt_filename, output_label_file):
    gt_path = os.path.join(data_dir, gt_filename)
    if not os.path.exists(gt_path):
        print(f"Error: File tidak ditemukan di {gt_path}")
        return

    print(f"Memproses file: {gt_path}")
    count = 0
    # Menggunakan 'utf-8-sig' untuk menghapus karakter BOM tak terlihat
    with open(gt_path, 'r', encoding='utf-8-sig') as f_in, \
        open(output_label_file, 'w', encoding='utf-8') as f_out:
        
        for line in f_in:
            try:
                parts = line.strip().split(', ')
                image_name = parts[0]
                label = parts[1].strip('"')

                # Membuat path lengkap yang benar, relatif dari folder proyek
                full_image_path = os.path.join(data_dir, image_name).replace('\\', '/')
                
                # --- PERBAIKAN FINAL DI SINI ---
                # Menulis dengan urutan yang benar: PATH dulu, baru LABEL
                f_out.write(f"{full_image_path}\t{label.lower()}\n")
                count += 1
            except IndexError:
                print(f"Peringatan: Melewati baris yang formatnya salah -> {line.strip()}")

    print(f"Selesai! Berhasil memproses {count} label ke file: {output_label_file}\n")


if __name__ == '__main__':
    kaggle_train_dir = 'ICDAR_Kaggle/Train'
    kaggle_test_dir = 'ICDAR_Kaggle/Test'
    
    process_gt_file(
        data_dir=kaggle_train_dir,
        gt_filename='gt.txt',
        output_label_file='dataset/kaggle_icdar_train_labels.txt'
    )
    process_gt_file(
        data_dir=kaggle_test_dir,
        gt_filename='gt.txt',
        output_label_file='dataset/kaggle_icdar_val_labels.txt'
    )