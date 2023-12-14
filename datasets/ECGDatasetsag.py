#!/usr/bin/python
# -*- coding:utf-8 -*-

import torch
from torch.utils.data import Dataset
from scipy.io import loadmat
import numpy as np
import os

def Resample(input_signal, src_fs, tar_fs):
    '''
    :param input_signal:输入信号
    :param src_fs:输入信号采样率
    :param tar_fs:输出信号采样率
    :return:输出信号
    '''
    if src_fs != tar_fs:
        dtype = input_signal.dtype
        audio_len = input_signal.shape[1]
        audio_time_max = 1.0 * (audio_len) / src_fs
        src_time = 1.0 * np.linspace(0, audio_len, audio_len) / src_fs
        tar_time = 1.0 * np.linspace(0, int(audio_time_max * tar_fs), int(audio_time_max * tar_fs)) / tar_fs
        for i in range(input_signal.shape[0]):
            if i == 0:
                output_signal = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                output_signal = output_signal.reshape(1, len(output_signal))
            else:
                tmp = np.interp(tar_time, src_time, input_signal[i, :]).astype(dtype)
                tmp = tmp.reshape(1, len(tmp))
                output_signal = np.vstack((output_signal, tmp))
    else:
        output_signal = input_signal
    return output_signal


def LPF_Resample(input_signal, src_fs, tar_fs, filter_order=4):
    """
    Resamples a signal using Butterworth low-pass filter and zero-phase digital
    filtering. If the input is a collection of signals, the signals are
    resampled individually.
    The resulting signal better approximates the original signal as compared
    to simple linear interpolation.
    Args:
        input_signal (1D or 2D NumPy array): The input signal.
        src_fs (float): The sampling frequency of the input signal.
        tar_fs (float): The desired frequency to resample the signal to.
    """
    if src_fs == tar_fs:
        return input_signal
    from scipy import signal
    if input_signal.ndim == 1:
        nyquist_freq = tar_fs / 2
        lpf = signal.butter(N=filter_order, Wn=nyquist_freq, btype='low', fs=src_fs, output='sos') # Low-pass filter
        lps = signal.sosfiltfilt(lpf, input_signal) # Low-passed signal
        rss_num_samples = int(np.round(input_signal.shape[0] / src_fs * tar_fs))
        rss = signal.resample(lps, num=rss_num_samples) # Resampled signal
        return rss
    
    elif input_signal.ndim == 2:
        # Recurse on each signal
        acc = []
        for signal in input_signal:
            acc.append(LPF_Resample(signal, src_fs, tar_fs, filter_order))
        return np.stack(acc)

    else:
        raise ValueError('LPF_Resample only supports input signal with ndim <= 2')


def load_data(case, src_fs, tar_fs=257, resample='butterworth'):
    x = loadmat(case)
    data = np.asarray(x['val'], dtype=np.float64)
    if resample=='butterworth':
        data = LPF_Resample(data, src_fs, tar_fs)
    else:
        data = Resample(data, src_fs, tar_fs)
    return data

def prepare_data(age, gender):
    data = np.zeros(5,)
    if age >= 0:
        data[0] = age / 100
    if 'F' in gender:
        data[2] = 1
        data[4] = 1
    elif gender == 'Unknown':
        data[4] = 0
    elif 'f' in gender:
        data[2] = 1
        data[4] = 1
    else:
        data[3] = 1
        data[4] = 1

    return data

class dataset(Dataset):

    def __init__(self, anno_pd, test=False, transform=None, data_dir=None, loader=load_data, resample='butterworth'):
        self.test = test
        if self.test:
            self.data = anno_pd['filename'].tolist()
            self.fs = anno_pd['fs'].tolist()
        else:
            self.data = anno_pd['filename'].tolist()
            labels = anno_pd.iloc[:, 4:].values
            self.multi_labels = [labels[i, :] for i in range(labels.shape[0])]
            self.age = anno_pd['age'].tolist()
            self.gender = anno_pd['gender'].tolist()
            self.fs = anno_pd['fs'].tolist()

        self.transforms = transform
        self.data_dir = data_dir
        self.loader = loader
        self.resample = resample


    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        if self.test:
            img_path = self.data[item]
            fs = self.fs[item]
            img = self.loader(self.data_dir + img_path, src_fs=fs, resample=self.resample)
            img = self.transforms(img)
            return img, img_path
        else:
            img_name = self.data[item]
            fs = self.fs[item]
            age = self.age[item]
            gender = self.gender[item]
            age_gender = prepare_data(age, gender)
            img = self.loader(img_name, src_fs=fs, resample=self.resample)
            label = self.multi_labels[item]
            """
            for i in range(img.shape[1]):
                img[:, i] = ecg_preprocessing(img[:, i], wfun='db6', levels=9, type=2)
            """
            img = self.transforms(img)
            return img, torch.from_numpy(age_gender).float(), torch.from_numpy(label).float()

if __name__ == '__main__':
    """
    img = cv2.imread('../ODIR-5K_training/0_left.jpg')
    #cv2.flip(img, 1, dst=None)
    cv2.namedWindow("resized", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("resized", 640, 480)
    cv2.imshow('resized', img)
    cv2.waitKey(5)
    # cv2.destoryAllWindows()
    """
