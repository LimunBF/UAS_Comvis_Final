import torch
import torch.nn as nn
import torch.nn.functional as F

class Attention(nn.Module):
    """
    Modul Attention yang sudah diperbaiki.
    """
    def __init__(self, rnn_feature_size, num_classes):
        super(Attention, self).__init__()
        # Attention cell mengambil fitur dari RNN untuk menghitung 'bobot perhatian'
        self.attention_cell = nn.Linear(rnn_feature_size, 1)
        
        # Generator (layer prediksi akhir) mengambil gabungan fitur dari RNN dan 'vektor perhatian'
        # Ukuran inputnya adalah rnn_feature_size (dari RNN) + rnn_feature_size (dari attention vector)
        self.generator = nn.Linear(rnn_feature_size * 2, num_classes)

    def forward(self, rnn_output):
        # rnn_output shape: [batch, width, rnn_feature_size]
        
        # Menghitung seberapa penting setiap 'timestep' dari RNN
        attention_energy = self.attention_cell(rnn_output)
        attention_weights = F.softmax(attention_energy.squeeze(2), dim=1).unsqueeze(2)
        
        # Membuat 'vektor perhatian' yang merupakan rangkuman tertimbang dari output RNN
        context_vector = torch.sum(rnn_output * attention_weights, dim=1).unsqueeze(1)
        attention_vector = context_vector.repeat(1, rnn_output.size(1), 1)

        # Menggabungkan informasi asli dari RNN dengan 'vektor perhatian'
        concatenated_output = torch.cat([rnn_output, attention_vector], dim=2)
        
        # Melakukan prediksi akhir menggunakan gabungan fitur yang lebih kaya
        prediction = self.generator(concatenated_output)
        return prediction

class CRNN_Attention(nn.Module):
    def __init__(self, img_height, num_channels, num_classes, rnn_hidden_size=256):
        super(CRNN_Attention, self).__init__()

        # Backbone CNN (Tidak ada perubahan)
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 64, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 256, 3, 1, 1), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.Conv2d(256, 256, 3, 1, 1), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(256, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True),
            nn.Conv2d(512, 512, 3, 1, 1, bias=False), nn.BatchNorm2d(512), nn.ReLU(True), nn.MaxPool2d((2, 1), (2, 1)),
            nn.AdaptiveAvgPool2d((1, None))
        )

        # RNN (Tidak ada perubahan)
        self.rnn = nn.LSTM(
            input_size=512,
            hidden_size=rnn_hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
            dropout=0.5
        )

        # --- PERBAIKAN INISIALISASI ATTENTION ---
        # Ukuran fitur dari RNN bidirectional adalah hidden_size * 2
        rnn_feature_size = rnn_hidden_size * 2
        self.attention = Attention(rnn_feature_size, num_classes)
        # ----------------------------------------

    def forward(self, x):
        conv_features = self.cnn(x)
        squeezed = conv_features.squeeze(2)
        permuted = squeezed.permute(0, 2, 1)
        
        rnn_out, _ = self.rnn(permuted)
        output = self.attention(rnn_out)
        
        return output