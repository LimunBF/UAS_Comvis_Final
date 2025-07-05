import torch
import torch.nn as nn
import torch.nn.functional as F

class CTCLossWithLabelSmoothing(nn.Module):
    def __init__(self, num_classes, blank=0, smoothing=0.1, reduction='mean'):
        """
        CTCLoss yang dimodifikasi dengan Label Smoothing.
        
        Args:
            num_classes (int): Jumlah kelas (termasuk blank).
            blank (int): Indeks untuk token blank.
            smoothing (float): Faktor smoothing (misal 0.1).
            reduction (str): 'mean' atau 'sum'.
        """
        super(CTCLossWithLabelSmoothing, self).__init__()
        self.ctc_loss = nn.CTCLoss(blank=blank, reduction=reduction, zero_infinity=True)
        self.num_classes = num_classes
        self.smoothing = smoothing
        self.confidence = 1.0 - smoothing
        self.reduction = reduction

    def forward(self, log_probs, targets, input_lengths, target_lengths):
        """
        log_probs: Output dari model setelah log_softmax [T, B, C]
        targets: Label sebenarnya [B, S]
        """
        # 1. Hitung loss CTC standar
        ctc_loss = self.ctc_loss(log_probs, targets, input_lengths, target_lengths)
        
        # 2. Hitung komponen Label Smoothing
        # Kita ingin menghukum model jika terlalu percaya diri
        smooth_loss = -log_probs.mean(dim=-1) # Rata-rata dari semua log probabilitas
        
        if self.reduction == 'mean':
            smooth_loss = smooth_loss.mean()
        elif self.reduction == 'sum':
            smooth_loss = smooth_loss.sum()
            
        # 3. Gabungkan keduanya
        # Sebagian besar loss berasal dari CTC, dan sebagian kecil dari smoothing
        total_loss = self.confidence * ctc_loss + self.smoothing * smooth_loss
        
        return total_loss