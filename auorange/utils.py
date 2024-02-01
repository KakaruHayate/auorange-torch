# Copyright 2020 Yablon Ding

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#    http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.io import wavfile


def load_wav(path, sample_rate):
  sr, raw_data = wavfile.read(path)
  if sample_rate != sr:
    raise ValueError('sample rate not equal')
  raw_data = raw_data.astype(np.float32)
  return (raw_data + 32768) / 65535. * 2 - 1


def save_wav(wav, path, sample_rate):
  data = (wav + 1) / 2 * 65535. - 32768
  wavfile.write(path, sample_rate, data.astype(np.int16))


def plot(array):
  fig = plt.figure(figsize=(30, 5))
  ax = fig.add_subplot(111)
  ax.xaxis.label.set_color('grey')
  ax.yaxis.label.set_color('grey')
  ax.xaxis.label.set_fontsize(23)
  ax.yaxis.label.set_fontsize(23)
  ax.tick_params(axis='x', colors='grey', labelsize=23)
  ax.tick_params(axis='y', colors='grey', labelsize=23)
  plt.plot(array)
  plt.show()


def plot_spec(M):
  M = np.flip(M, axis=0)
  plt.figure(figsize=(18, 4))
  plt.imshow(M, interpolation='nearest', aspect='auto')
  plt.show()
