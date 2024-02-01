import matplotlib.pyplot as plt
import torch
import numpy as np


from auorange.utils import plot, plot_spec, load_wav, save_wav
from auorange.auorange_torch import Audio2Mel,Mel2LPC, LPC2Wav, LPC2WavWithResidual

wav_name = 'wavs/2001000006.wav'
sample_rate = 44100
n_fft = 2048
num_mels = 128
hop_length = 512
win_length = 2048
lpc_order = 8
clip_lpc = True
mel_fmin = 40
f0 = 40.

wav_data = load_wav(wav_name, sample_rate)
wav_data = torch.tensor(wav_data).unsqueeze(0).unsqueeze(1)


a2w = Audio2Mel(
    sampling_rate=sample_rate, 
    hop_length=hop_length, 
    win_length=win_length, 
    n_fft=n_fft, 
    n_mel_channels=num_mels, 
    mel_fmin=mel_fmin, 
    mel_fmax=None
    )
mel = a2w(wav_data)


m2l = Mel2LPC(
    sampling_rate=sample_rate, 
    hop_length=hop_length, 
    win_length=win_length, 
    n_fft=n_fft, 
    n_mel_channels=num_mels, 
    mel_fmin=mel_fmin, 
    mel_fmax=None, 
    f0=f0, 
    lpc_order=lpc_order
    )
LPC_ctrl = m2l(mel)


l2w = LPC2Wav(lpc_order=lpc_order, clip_lpc=True)
wav_pred = l2w(LPC_ctrl,wav_data)

# Make sure the predicted audio and the original audio have the same shape
if wav_pred.shape[2] > wav_data.shape[2]:
    wav_pred = wav_pred[:, :, :wav_data.shape[2]]
elif wav_pred.shape[2] < wav_data.shape[2]:
    wav_data = wav_data[:, :, :wav_pred.shape[2]]

# Compute the residual
residual = wav_data - wav_pred


r2w = LPC2WavWithResidual(lpc_order=lpc_order, clip_lpc=True)
error_pred = r2w(LPC_ctrl, residual)


wav_data = wav_data.squeeze(0).squeeze(0).numpy()
wav_pred = wav_pred.squeeze(0).squeeze(0).numpy()
error_pred = error_pred.squeeze(0).squeeze(0).numpy()
error = residual.squeeze(0).squeeze(0).numpy()


save_wav(wav_pred, 'wavs/wav_pred2001000006.wav', sample_rate)
save_wav(error, 'wavs/error2001000006.wav', sample_rate)
save_wav(error_pred, 'wavs/error_pred2001000006.wav', sample_rate)


fig = plt.figure(figsize=(30, 5))
plt.subplot(311)
plt.ylabel('wav_data')
plt.xlabel('time')
plt.plot(wav_data)
plt.subplot(312)
plt.ylabel('wav_pred')
plt.xlabel('time')
plt.plot(wav_pred)
plt.subplot(313)
plt.ylabel('error')
plt.xlabel('time')
plt.plot(error)
plt.show()
