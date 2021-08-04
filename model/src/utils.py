"""
author: Wira D K Putra
25 February 2020

See original repo at
https://github.com/WiraDKP/pytorch_speaker_embedding_for_diarization
"""

import torch
import librosa
import numpy as np
from itertools import groupby
from scipy.ndimage import gaussian_filter1d


def zcr_vad(y, shift=0.025, win_len=2048, hop_len=1024, threshold=0.005):
    if isinstance(y, torch.Tensor):
        y = y.cpu().numpy()
    if y.ndim == 2:
        y = y[0]
    zcr = librosa.feature.zero_crossing_rate(y + shift, win_len, hop_len)[0]
    activity = gaussian_filter1d(zcr, 1) > threshold
    activity = np.repeat(activity, len(y) // len(activity) + 1)
    activity = activity[:len(y)]
    return activity


def get_timestamp(activity):
    mask = [k for k, _ in groupby(activity)]
    #print(mask, "mask==============>") # false, true 
    #print(activity.shape, "active_shape==================>")
    change = np.argwhere(activity[:-1] != activity[1:]).flatten() #[:-1]除了最后一个取全部, [1:]取第二个到最后一个元素
    #print(change.shape, "change.shape===============>") # array
    span = np.concatenate([[0], change, [len(activity)]])
    #print(span, "span_concatenate=====================>")
    span = list(zip(span[:-1], span[1:]))
    #print(span, "span_List================>")
    span = np.array(span)[mask]
    #print(span, "span_nparray=============>") #調整維度
    return span
