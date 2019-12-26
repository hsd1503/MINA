import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle as dill
import sys
import os
import random
from tqdm import tqdm
from collections import OrderedDict, Counter

from sklearn.metrics import roc_auc_score, average_precision_score, f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import scipy.io
from scipy.signal import butter, lfilter, periodogram


def preprocess_physionet(data_path):
    """
    before running this data preparing code, 
    please first download the raw data from https://physionet.org/content/challenge-2017/1.0.0/, 
    and put it in data_path
    """
    
    # read label
    label_df = pd.read_csv(os.path.join(data_path, 'REFERENCE-v3.csv'), header=None)
    label = label_df.iloc[:,1].values
    print(Counter(label))

    # read data
    all_data = []
    filenames = pd.read_csv(os.path.join(data_path, 'RECORDS'), header=None)
    filenames = filenames.iloc[:,0].values
    print(filenames)
    for filename in tqdm(filenames):
        mat = scipy.io.loadmat(os.path.join(data_path, 'training2017/{0}.mat'.format(filename)))
        mat = np.array(mat['val'])[0]
        all_data.append(mat)
    all_data = np.array(all_data)

    res = {'data':all_data, 'label':label}
    with open(os.path.join(data_path, 'challenge2017.pkl'), 'wb') as fout:
        dill.dump(res, fout)

def filter_channel(x):
    
    signal_freq = 300
    
    ### candidate channels for ECG
    P_wave = (0.67,5)
    QRS_complex = (10,50)
    T_wave = (1,7)
    muscle = (5,50)
    resp = (0.12,0.5)
    ECG_preprocessed = (0.5, 50)
    wander = (0.001, 0.5)
    noise = 50
    
    ### use low (wander), middle (ECG_preprocessed) and high (noise) for example
    bandpass_list = [wander, ECG_preprocessed]
    highpass_list = [noise]
    
    nyquist_freq = 0.5 * signal_freq
    filter_order = 1
    ### out including original x
    out_list = [x]
    
    for bandpass in bandpass_list:
        low = bandpass[0] / nyquist_freq
        high = bandpass[1] / nyquist_freq
        b, a = butter(filter_order, [low, high], btype="band")
        y = lfilter(b, a, x)
        out_list.append(y)
        
    for highpass in highpass_list:
        high = highpass / nyquist_freq
        b, a = butter(filter_order, high, btype="high")
        y = lfilter(b, a, x)
        out_list.append(y)
        
    out = np.array(out_list)
    
    return out

def slide_and_cut(X, Y, window_size, stride, output_pid=False):
    out_X = []
    out_Y = []
    out_pid = []
    n_sample = X.shape[0]
    mode = 0
    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]
        if tmp_Y == 0:
            i_stride = stride
        elif tmp_Y == 1:
            i_stride = stride//10
        for j in range(0, len(tmp_ts)-window_size, i_stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(tmp_Y)
            out_pid.append(i)
    if output_pid:
        return np.array(out_X), np.array(out_Y), np.array(out_pid)
    else:
        return np.array(out_X), np.array(out_Y)

def compute_beat(X):
    out = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in tqdm(range(out.shape[0]), desc="compute_beat"):
        for j in range(out.shape[1]):
            out[i, j] = np.concatenate([[0], np.abs(np.diff(X[i,j,:]))])
    return out

def compute_rhythm(X, n_split):
    cnt_split = int(X.shape[2]/n_split)
    out = np.zeros((X.shape[0], X.shape[1], cnt_split))
    for i in tqdm(range(out.shape[0]), desc="compute_rhythm"):
        for j in range(out.shape[1]):
            tmp_ts = X[i,j,:]
            tmp_ts_cut = np.split(tmp_ts, X.shape[2]/n_split)
            for k in range(cnt_split):
                out[i, j, k] = np.std(tmp_ts_cut[k])
    return out

def compute_freq(X):
    out = np.zeros((X.shape[0], X.shape[1], 1))
    fs = 300
    for i in tqdm(range(out.shape[0]), desc="compute_freq"):
        for j in range(out.shape[1]):
            _, Pxx_den = periodogram(X[i,j,:], fs)
            out[i, j, 0] = np.sum(Pxx_den)
    return out

def make_data_physionet(data_path, n_split=50, window_size=3000, stride=500):

    # read pkl
    with open(os.path.join(data_path, 'challenge2017.pkl'), 'rb') as fin:
        res = dill.load(fin)
    ## scale data
    all_data = res['data']
    for i in range(len(all_data)):
        tmp_data = all_data[i]
        tmp_std = np.std(tmp_data)
        tmp_mean = np.mean(tmp_data)
        all_data[i] = (tmp_data - tmp_mean) / tmp_std # normalize
    all_data = res['data']
    all_data = np.array(all_data)
    ## encode label
    all_label = []
    for i in res['label']:
        if i == 'A':
            all_label.append(1)
        else:
            all_label.append(0)
    all_label = np.array(all_label)

    # split train test
    n_sample = len(all_label)
    split_idx_1 = int(0.75 * n_sample)
    split_idx_2 = int(0.85 * n_sample)
    
    shuffle_idx = np.random.permutation(n_sample)
    all_data = all_data[shuffle_idx]
    all_label = all_label[shuffle_idx]
    
    X_train = all_data[:split_idx_1]
    X_val = all_data[split_idx_1:split_idx_2]
    X_test = all_data[split_idx_2:]
    Y_train = all_label[:split_idx_1]
    Y_val = all_label[split_idx_1:split_idx_2]
    Y_test = all_label[split_idx_2:]
    
    # slide and cut
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=window_size, stride=stride)
    X_val, Y_val = slide_and_cut(X_val, Y_val, window_size=window_size, stride=stride)
    X_test, Y_test, pid_test = slide_and_cut(X_test, Y_test, window_size=window_size, stride=stride, output_pid=True)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    
    # shuffle train
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]

    # multi-level
    X_train_ml = []
    X_val_ml = []
    X_test_ml = []
    for i in tqdm(X_train, desc="X_train_ml"):
        tmp = filter_channel(i)
        X_train_ml.append(tmp)
    X_train_ml = np.array(X_train_ml)
    for i in tqdm(X_val, desc="X_val_ml"):
        tmp = filter_channel(i)
        X_val_ml.append(tmp)
    X_val_ml = np.array(X_val_ml)
    for i in tqdm(X_test, desc="X_test_ml"):
        tmp = filter_channel(i)
        X_test_ml.append(tmp)
    X_test_ml = np.array(X_test_ml)
    print(X_train_ml.shape, X_val_ml.shape, X_test_ml.shape)

    # save
    res = {'Y_train': Y_train, 'Y_val': Y_val, 'Y_test': Y_test, 'pid_test': pid_test}
    with open(os.path.join(data_path, 'mina_info.pkl'), 'wb') as fout:
        dill.dump(res, fout)
        
    fout = open(os.path.join(data_path, 'mina_X_train.bin'), 'wb')
    np.save(fout, X_train_ml)
    fout.close()

    fout = open(os.path.join(data_path, 'mina_X_val.bin'), 'wb')
    np.save(fout, X_val_ml)
    fout.close()

    fout = open(os.path.join(data_path, 'mina_X_test.bin'), 'wb')
    np.save(fout, X_test_ml)
    fout.close()

def make_knowledge_physionet(data_path, n_split=50):

    # read
    fin = open(os.path.join(data_path, 'mina_X_train.bin'), 'rb')
    X_train = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_X_val.bin'), 'rb')
    X_val = np.load(fin)
    fin.close()
    fin = open(os.path.join(data_path, 'mina_X_test.bin'), 'rb')
    X_test = np.load(fin)
    fin.close()

    # compute knowledge
    K_train_beat = compute_beat(X_train)
    K_train_rhythm = compute_rhythm(X_train, n_split)
    K_train_freq = compute_freq(X_train)

    K_val_beat = compute_beat(X_val)
    K_val_rhythm = compute_rhythm(X_val, n_split)
    K_val_freq = compute_freq(X_val)

    K_test_beat = compute_beat(X_test)
    K_test_rhythm = compute_rhythm(X_test, n_split)
    K_test_freq = compute_freq(X_test)

    # save
    fout = open(os.path.join(data_path, 'mina_K_train_beat.bin'), 'wb')
    np.save(fout, K_train_beat)
    fout.close()
    fout = open(os.path.join(data_path, 'mina_K_val_beat.bin'), 'wb')
    np.save(fout, K_val_beat)
    fout.close()
    fout = open(os.path.join(data_path, 'mina_K_test_beat.bin'), 'wb')
    np.save(fout, K_test_beat)
    fout.close()

    res = {'K_train_rhythm': K_train_rhythm, 'K_train_freq': K_train_freq, 
    'K_val_rhythm': K_val_rhythm, 'K_val_freq': K_val_freq, 
    'K_test_rhythm': K_test_rhythm, 'K_test_freq': K_test_freq}
    with open(os.path.join(data_path, 'mina_knowledge.pkl'), 'wb') as fout:
        dill.dump(res, fout)

def evaluate(gt, pred):
    '''
    gt is (0, C-1)
    pred is list of probability
    '''

    pred_label = []
    for i in pred:
        pred_label.append(np.argmax(i))
    pred_label = np.array(pred_label)

    res = OrderedDict({})
    
    res['auroc'] = roc_auc_score(gt, pred[:,1])
    res['auprc'] = average_precision_score(gt, pred[:,1])
    res['f1'] = f1_score(gt, pred_label)
    
    res['\nmat'] = confusion_matrix(gt, pred_label)
    
    for k, v in res.items():
        print(k, ':', v, '|', end='')
    print()
    
    return list(res.values())
