import os
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


class PreEmphasis(torch.nn.Module):
    def __init__(self, coefficient: float = 0.9375):
        super().__init__()
        self.coefficient = coefficient
        self.register_buffer('kernel', torch.tensor([-coefficient, 1.], dtype=torch.float32).unsqueeze(0).unsqueeze(0))

    def forward(self, signal):
        return F.conv1d(signal, self.kernel)


class Audio2Mel(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, clamp=1e-5, cpt_lpc=True, mel_base='e'):
        super().__init__()

        n_fft = win_length if n_fft is None else n_fft

        self.mel_base = mel_base
        self.cpt_lpc = cpt_lpc
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.clamp = clamp

        # get mel_basis
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, 
                                   fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        self.register_buffer("mel_basis", mel_basis)
        self.n_mel_channels = n_mel_channels
        self.hann_window = {}


    def forward(self, audio, keyshift=0, speed=1):
        '''
        Input:
            audio: [B, 1, T]
        Returns:
            log_mel_spec: [B, M, T] 
        '''
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        B, C, T = audio.shape
        audio = audio.reshape(B * C, T)
        fft = torch.stft(audio, n_fft=n_fft_new, hop_length=hop_length_new,
                         win_length=win_length_new, window=self.hann_window[keyshift_key],
                         center=True, return_complex=True)
        magnitude = torch.abs(fft)
        
        if keyshift != 0:
            size = self.n_fft // 2 + 1
            resize = magnitude.size(1)
            if resize < size:
                magnitude = F.pad(magnitude, (0, 0, 0, size-resize))
            magnitude = magnitude[:, :size, :] * self.win_length / win_length_new
            
        mel_output = torch.matmul(self.mel_basis, magnitude)
        if self.mel_base == 'ori':
            # In the original repo, the mel base number was set as follows:
            log_mel_spec = torch.log(1. + 10000 * mel_output) 
        elif self.mel_base == '10':
            # At a certain period, the mel base number is 10
            log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))
        elif self.mel_base == 'e':
            # Nowadays, mel base number is set as the natural logarithm e (which is actually equivalent to the most primitive repo)
            log_mel_spec = torch.log(torch.clamp(mel_output, min=self.clamp))
        else:
            raise ValueError("Unsupported mel base type")
        
        # log_mel_spec: B x C, M, T
        T_ = log_mel_spec.shape[-1]
        log_mel_spec = log_mel_spec.reshape(B, C, self.n_mel_channels ,T_)
        log_mel_spec = log_mel_spec.permute(0, 3, 1, 2)

        # print('og_mel_spec:', log_mel_spec.shape)
        log_mel_spec = log_mel_spec.squeeze(2) # mono

        return log_mel_spec


class Mel2LPC(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, repeat=None, f0=40., 
                 lpc_order=4, clamp = 1e-12, mel_base='e'):
        super().__init__()

        n_fft = win_length if n_fft is None else n_fft
        repeat = hop_length if repeat is None else repeat

        self.mel_base = mel_base
        self.sampling_rate = sampling_rate
        self.hop_length = hop_length
        self.win_length = win_length
        self.n_fft = n_fft
        self.repeat = repeat
        self.clamp = clamp

        self.register_buffer("n_mel_channels", torch.tensor(n_mel_channels))
        self.register_buffer("lpc_order", torch.tensor(lpc_order))

        # get lag_window
        theta = (2 * torch.pi * f0 / self.sampling_rate)**2
        self.register_buffer("lag_window", torch.exp(-0.5 * theta * torch.arange(self.lpc_order + 1).type(torch.float32)**2).unsqueeze(1))

        # get inv_mel_basis
        mel_basis = librosa_mel_fn(sr=sampling_rate, n_fft=n_fft, n_mels=n_mel_channels, 
                                   fmin=mel_fmin, fmax=mel_fmax)
        mel_basis = torch.from_numpy(mel_basis).float()
        inv_mel_basis = torch.pinverse(mel_basis)
        self.register_buffer("inv_mel_basis", inv_mel_basis)
        
        
    def LevinsonDurbin(self, pAC):
        """levinson durbin's recursion
        Input:
            pAC (tensor): autocorrelation [1, n+1, T]

        Returns:
            tensor: lpc coefficients [1, n, T]
        """
        _, n_plus_1, num_frames = pAC.shape
        n = n_plus_1 - 1 # Lpc_order is the dimension of the autocorrelation coefficient minus 1
        pLP = torch.zeros([1, n, num_frames], dtype=torch.float32)
        pTmp = torch.zeros([n, num_frames], dtype=torch.float32)
        E = pAC[0, 0, :].clone()

        for i in range(n):
            ki = pAC[0, i + 1, :] + torch.sum(pLP[0, :i, :] * pAC[0, i - torch.arange(i), :], dim=0)
            ki /= E
            c = (1 - ki * ki).clamp(min=1e-5)
            E *= c
            pTmp[i, :] = -ki
            for j in range(i):
                pTmp[j, :] = pLP[0, j, :] - ki * pLP[0, i - j - 1, :]
            pLP[0, :i, :] = pTmp[:i, :]
            pLP[0, i, :] = pTmp[i, :]


        return pLP


    def forward(self, mel):
        '''
        Input:
            mel: [1, M, T]
        Returns:
            lpc_ctrl: [1, lpc_order, T]
        '''
        # mel to linear
        if self.mel_base == 'ori':
            mel = (torch.exp(mel) - 1.) / 10000
        elif self.mel_base == 'e':
            mel = torch.exp(mel)
        elif self.mel_base == '10':
            mel = torch.pow(10, mel)
        else:
            raise ValueError("Unsupported mel base type")
        
        linear = torch.clamp_min(torch.matmul(self.inv_mel_basis, mel), self.clamp)

        # linear to autocorrelation
        power = linear**2
        flipped_power = torch.flip(power, dims=[1])[:,1:-1, :]
        fft_power = torch.cat([power, flipped_power], dim=1)
        auto_correlation = torch.fft.ifft(fft_power, dim=1).real
        
        # autocorrelation to lpc
        auto_correlation = auto_correlation[:, 0:self.lpc_order + 1, :]
        auto_correlation = auto_correlation * self.lag_window
        lpc_ctrl = self.LevinsonDurbin(auto_correlation)
        lpc_ctrl = -1 * torch.flip(lpc_ctrl, dims=[1])
        if self.repeat is not None:
            lpc_ctrl = torch.repeat_interleave(lpc_ctrl, self.repeat, dim=-1)


        return lpc_ctrl


def LPC2Wav(lpc_ctrl, wav, lpc_order, clip_lpc):
    '''
    lpc_ctrl: [1, lpc_order, T]
    wav: [B, C, T]
    lpc_order: int
    clip_lpc:bool

    pred: [B, C, T]
    '''
    lpc_ctrl = lpc_ctrl[:, :, :wav.shape[-1]]
    num_points = lpc_ctrl.shape[-1]
    if wav.shape[2] == num_points:
        wav = F.pad(wav, (lpc_order, 0), 'constant')
    elif wav.shape[2] != num_points + lpc_order:
        raise RuntimeError('dimensions of lpcs and audio must match')

    indices = (torch.arange(lpc_order).view(-1, 1) + torch.arange(lpc_ctrl.shape[-1]))
    signal_slices = wav[:, :, indices]

    # predict
    pred = torch.sum(lpc_ctrl * signal_slices, dim=2)
    if clip_lpc:
        pred = torch.clip(pred, -1., 1.)


    return pred
