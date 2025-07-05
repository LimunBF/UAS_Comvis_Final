import torch

class LabelEncoder:
    def __init__(self, charset):
        """
        Inisialisasi LabelEncoder dengan set karakter yang diberikan.
        
        Args:
            charset (str): String yang berisi semua karakter unik dalam dataset.
        """
        self.charset = charset
        
        # --- Bagian Penting yang Hilang ---
        # Membuat kamus untuk mapping karakter ke integer dan sebaliknya.
        # Indeks 0 biasanya dicadangkan untuk 'blank' token CTC.
        self.char_to_int = {char: i + 1 for i, char in enumerate(self.charset)}
        self.int_to_char = {i + 1: char for i, char in enumerate(self.charset)}

    def encode(self, text):
        """
        Meng-encode sebuah string teks menjadi urutan integer.
        Karakter yang tidak ada di dalam charset akan diabaikan.
        
        Mengembalikan tuple: (urutan integer, panjang urutan).
        """
        encoded_text = [self.char_to_int[char] for char in text if char in self.char_to_int]
        return encoded_text, len(encoded_text)

    def decode(self, tokens):
        """
        Mendekode urutan integer (hasil dari prediksi model) kembali menjadi string.
        Token 'blank' (0) dan duplikat akan diabaikan.
        """
        # CTC Decode sederhana: hapus duplikat berurutan dan token blank (0)
        decoded_chars = []
        for i, token in enumerate(tokens):
            if token != 0 and (i == 0 or token != tokens[i - 1]):
                if token in self.int_to_char:
                    decoded_chars.append(self.int_to_char[token])
        return "".join(decoded_chars)