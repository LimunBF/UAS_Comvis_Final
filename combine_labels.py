import os

# File-file label yang ingin digabungkan
file_list = [
    'dataset/train_labels.txt',              # Label training asli dari IIIT-5K
    'dataset/kaggle_icdar_train_labels.txt'  # Label training baru dari Kaggle
]

# File output gabungan
output_file = 'dataset/combined_train_labels.txt'

all_lines = []
for file_path in file_list:
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            all_lines.extend(lines)
            print(f"Membaca {len(lines)} baris dari {file_path}")
    else:
        print(f"Peringatan: File tidak ditemukan di {file_path}")

with open(output_file, 'w', encoding='utf-8') as f_out:
    f_out.writelines(all_lines)

print(f"\nBerhasil menggabungkan semua label ke dalam: {output_file}")
print(f"Total kata untuk training sekarang: {len(all_lines)}")