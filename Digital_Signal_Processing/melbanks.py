from typing import Optional

import torch
from torch import nn
from torchaudio import functional as F


class LogMelFilterBanks(nn.Module):
    def __init__(
            self,
            n_fft: int = 400,
            samplerate: int = 16000,
            hop_length: int = 160,
            n_mels: int = 80,
            pad_mode: str = 'reflect',
            power: float = 2.0,
            normalize_stft: bool = False,
            onesided: bool = True,
            center: bool = True,
            return_complex: bool = True,
            f_min_hz: float = 0.0,
            f_max_hz: Optional[float] = None,
            norm_mel: Optional[str] = None,
            mel_scale: str = 'htk'
        ):
        super(LogMelFilterBanks, self).__init__()
        # general params and params defined by the exercise
        self.n_fft = n_fft
        self.samplerate = samplerate
        self.window_length = n_fft
        self.window = torch.hann_window(self.window_length)
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.center = center 
        self.return_complex = return_complex
        self.onesided = onesided
        self.normalize_stft = normalize_stft
        self.pad_mode = pad_mode
        self.power = power
        self.f_min_hz = f_min_hz
        self.f_max_hz = f_max_hz
        if self.f_max_hz is None:
            self.f_max_hz = samplerate / 2.0
        self.norm_mel = norm_mel
        self.mel_scale = mel_scale
        self.mel_fbanks = self._init_melscale_fbanks()
        self.register_buffer('eps', torch.tensor(1e-6))

    def _init_melscale_fbanks(self):
        return F.melscale_fbanks(
            # Turns a normal STFT into a mel frequency STFT with triangular filter banks
            n_freqs = self.n_fft // 2 + 1,
            f_min = self.f_min_hz,
            f_max = self.f_max_hz,
            n_mels = self.n_mels,
            sample_rate = self.samplerate,
            norm = self.norm_mel,
            mel_scale = self.mel_scale
        )

    def spectrogram(self, x):
        # x - is an input signal
        if x.dim() == 3:
            x = x.squeeze(1)
        elif x.dim() == 1:
            x = x.unsqueeze(0)
        return torch.stft(
            x,
            n_fft = self.n_fft,
            hop_length = self.hop_length,
            win_length = self.window_length,
            window = self.window,
            center = self.center,
            pad_mode = self.pad_mode,
            return_complex = self.return_complex
        )

    def forward(self, x):
        """
        Args:
            x (Torch.Tensor): Tensor of audio of dimension (batch, time), audiosignal
        Returns:
            Torch.Tensor: Tensor of log mel filterbanks of dimension (batch, n_mels, n_frames),
                where n_frames is a function of the window_length, hop_length and length of audio
        """
        complex_spectrogram = self.spectrogram(x)

        if self.return_complex:
            power_spectrogram = complex_spectrogram.abs() ** self.power
        else:
            power_spectrogram = complex_spectrogram ** self.power
        
        # Applying mel-filters
        mel_spectrogram = torch.matmul(power_spectrogram.transpose(-2, -1), self.mel_fbanks).transpose(-2, -1)
        log_mel_spectrogram = torch.log(mel_spectrogram + self.eps)
        # Return log mel filterbanks matrix
        return log_mel_spectrogram
        

