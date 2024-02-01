import os
import numpy as np
import torch
import torch.nn.functional as F
from librosa.filters import mel as librosa_mel_fn


class LevinsonDurbin(torch.nn.Module):
    """levinson durbin's recursion
    Args:
        n (int): lpc order
        pAC (tensor): autocorrelation [1, n+1, T]

    Returns:
        tensor: lpc coefficients [1, n, T]
    """
    def __init__(self, n):
        super(LevinsonDurbin, self).__init__()
        self.n = int(n)

    def forward(self, pAC):
        num_frames = pAC.shape[-1]
        pLP = torch.zeros([1, self.n, num_frames], dtype=torch.float32)
        pTmp = torch.zeros([self.n, num_frames], dtype=torch.float32)
        E = pAC[0, 0, :].clone()

        # 
        for i in range(self.n):
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


class Audio2Mel(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, clamp=1e-5):
        super().__init__()

        n_fft = win_length if n_fft is None else n_fft

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

        self.register_buffer("n_mel_channels", torch.tensor(n_mel_channels))

        self.hann_window = {}


    def forward(self, audio, keyshift=0, speed=1):
        '''
        Args:
            audio: B x C x T
        Returns:
            log_mel_spec: B x n_mel x T 
        '''
        factor = 2 ** (keyshift / 12)       
        n_fft_new = int(np.round(self.n_fft * factor))
        win_length_new = int(np.round(self.win_length * factor))
        hop_length_new = int(np.round(self.hop_length * speed))
        
        keyshift_key = str(keyshift)+'_'+str(audio.device)
        if keyshift_key not in self.hann_window:
            self.hann_window[keyshift_key] = torch.hann_window(win_length_new).to(audio.device)
            
        B, C, T = audio.shape
        if C > 1:
            raise ValueError("Unsupported number of channels. Only mono (1) is allowed.")

        audio = audio.reshape(B, T)
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
        # DDSP里面用的是log10，但是这个项目原先用的是log(1. + 10000 * mel_output)
        log_mel_spec = torch.log10(torch.clamp(mel_output, min=self.clamp))
        #log_mel_spec = torch.log(1. + 10000 * mel_output)


        return log_mel_spec


class Mel2LPC(torch.nn.Module):
    def __init__(self, sampling_rate, hop_length, win_length, n_fft=None, 
                 n_mel_channels=128, mel_fmin=0, mel_fmax=None, repeat=None, f0=40., 
                 lpc_order=4, clamp = 1e-12 ):
        super().__init__()

        n_fft = win_length if n_fft is None else n_fft
        repeat = hop_length if repeat is None else repeat

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


    def forward(self, mel):
        '''
        Args:
            mel: B x n_mel x T
        Returns:
            LPC_ctrl: B x lpc_order x T
        '''
        # mel_to_linear
        mel = torch.pow(10, mel)
        #mel = (torch.exp(mel) - 1.) / 10000
        linear = torch.clamp_min(torch.matmul(self.inv_mel_basis, mel), self.clamp)

        # linear_to_autocorrelation
        power = linear**2
        flipped_power = torch.flip(power, dims=[1])[:,1:-1, :]
        fft_power = torch.cat([power, flipped_power], dim=1)
        auto_correlation = torch.fft.ifft(fft_power, dim=1).real

        # autocorrelation_to_lpc
        auto_correlation = auto_correlation[:, 0:self.lpc_order + 1, :]
        auto_correlation = auto_correlation * self.lag_window

        levinson_durbin = LevinsonDurbin(self.lpc_order)
        LD_jit = torch.jit.script(levinson_durbin)
        LPC_ctrl = LD_jit(auto_correlation)
        LPC_ctrl = -1 * torch.flip(LPC_ctrl, dims=[1])
        if self.repeat is not None:
            LPC_ctrl = torch.repeat_interleave(LPC_ctrl, self.repeat, dim=-1)


        return LPC_ctrl


class LPC2Wav(torch.nn.Module):
    def __init__(self, lpc_order=4, clip_lpc=True):
        super().__init__()

        self.clip_lpc = clip_lpc
        self.register_buffer("lpc_order", torch.tensor(lpc_order))


    def forward(self, LPC_ctrl, wav):
        '''
        LPC_ctrl: B x lpc_order x T
        wav: B x 1 x T

        pred: B x 1 x T
        '''
        LPC_ctrl = LPC_ctrl[:, :, :wav.shape[-1]]
        num_points = LPC_ctrl.shape[-1]
        if wav.shape[2] == num_points:
            wav = F.pad(wav, (self.lpc_order, 0), 'constant')
        elif wav.shape[2] != num_points + self.lpc_order:
            raise RuntimeError('dimensions of lpcs and audio must match')

        indices = (torch.arange(self.lpc_order).view(-1, 1) + torch.arange(LPC_ctrl.shape[-1]))
        signal_slices = wav[:, :, indices]

        # predict
        pred = torch.sum(LPC_ctrl * signal_slices, dim=2)
        if self.clip_lpc:
            pred = torch.clip(pred, -1., 1.)


        return pred


class LPC2WavWithResidual(torch.nn.Module):
    def __init__(self, lpc_order=4, clip_lpc=True):
        super().__init__()

        self.clip_lpc = clip_lpc
        self.register_buffer("lpc_order", torch.tensor(lpc_order))

    def forward(self, LPC_ctrl, residual):
        '''
        LPC_ctrl: B x lpc_order x T
        residual: B x 1 x T

        reconstructed: B x 1 x T
        '''
        LPC_ctrl = LPC_ctrl[:, :, :residual.shape[-1]]
        num_points = LPC_ctrl.shape[-1]
        if residual.shape[2] == num_points:
            residual = F.pad(residual, (self.lpc_order, 0), 'constant')
        elif residual.shape[2] != num_points + self.lpc_order:
            raise RuntimeError('dimensions of lpcs and residual must match')

        indices = (torch.arange(self.lpc_order).view(-1, 1) + torch.arange(LPC_ctrl.shape[-1]))
        signal_slices = residual[:, :, indices]

        # predict
        pred = torch.sum(LPC_ctrl * signal_slices, dim=2)
        if self.clip_lpc:
            pred = torch.clip(pred, -1., 1.)

        # add residual
        reconstructed = pred + residual[:, :, self.lpc_order:]

        return reconstructed
