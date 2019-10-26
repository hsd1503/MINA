import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import dill
import sys
import os
from collections import Counter
import random

from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import log_loss
from sklearn.metrics import confusion_matrix
from collections import OrderedDict

from scipy.signal import butter, lfilter

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

def slide_and_cut(X, Y, window_size):

    out_X = []
    out_Y = []
    n_sample = X.shape[0]
    mode = 0

    for i in range(n_sample):
        tmp_ts = X[i]
        tmp_Y = Y[i]

        if mode == 0:
            ### slide to get more training samples
            stride = 100
        else:
            ### or we can just select 1 sample from each, which is more likely to be clean signal
            stride = int(len(tmp_ts) - window_size/2)

        for j in range(0, len(tmp_ts)-window_size, stride):
            out_X.append(tmp_ts[j:j+window_size])
            out_Y.append(Y[i])
    
    return np.array(out_X), np.array(out_Y)

def make_challenge_data():
    
    n_dim = 3000

    ##################################################################
    ### read csv
    ##################################################################
    all_label = []
    all_data = []
    with open('data/long.csv', 'r') as fin:
        for line in fin:
            content = line.strip().split(',')
            pid = content[0]
            label = content[1]
            data = np.array([float(i) for i in content[2:]])
                
            if label == 'A':
                all_label.append(1)
            else:
                all_label.append(0)
            all_data.append(data)

    ##################################################################
    ### split train test
    ##################################################################
    all_data = np.array(all_data)
    all_label = np.array(all_label)
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
    
    ##################################################################
    ### slide and cut
    ##################################################################
    print('before: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    X_train, Y_train = slide_and_cut(X_train, Y_train, window_size=n_dim)
    X_val, Y_val = slide_and_cut(X_val, Y_val, window_size=n_dim)
    X_test, Y_test = slide_and_cut(X_test, Y_test, window_size=n_dim)
    print('after: ')
    print(Counter(Y_train), Counter(Y_val), Counter(Y_test))
    
    ##################################################################
    ### shuffle train
    ##################################################################
    shuffle_pid = np.random.permutation(Y_train.shape[0])
    X_train = X_train[shuffle_pid]
    Y_train = Y_train[shuffle_pid]
    
    # ##################################################################
    # ### multi-level
    # ##################################################################
    X_train_ml = []
    X_val_ml = []
    X_test_ml = []
    for i in X_train:
        tmp = filter_channel(i)
        X_train_ml.append(tmp)
    X_train_ml = np.array(X_train_ml)
    for i in X_val:
        tmp = filter_channel(i)
        X_val_ml.append(tmp)
    X_val_ml = np.array(X_val_ml)
    for i in X_test:
        tmp = filter_channel(i)
        X_test_ml.append(tmp)
    X_test_ml = np.array(X_test_ml)
    print(X_train_ml.shape, X_val_ml.shape, X_test_ml.shape)
    
    ##################################################################
    ### save
    ##################################################################
    res = {'X_train': X_train, 'Y_train': Y_train, 
           'X_val': X_val, 'Y_val': Y_val, 
           'X_test': X_test, 'Y_test': Y_test}
    
    with open('data/label.pkl', 'wb') as fout:
        dill.dump(res, fout)
        
    fout = open('data/X_train.bin', 'wb')
    np.save(fout, X_train_ml)
    fout.close()

    fout = open('data/X_val.bin', 'wb')
    np.save(fout, X_val_ml)
    fout.close()

    fout = open('data/X_test.bin', 'wb')
    np.save(fout, X_test_ml)
    fout.close()

def evaluate(gt, pred):
    '''
    gt is (0, C-1)
    pred is list of probability
    '''

    pred_label = []
    for i in pred:
        pred_label.append(np.argmax(i))
        
    gt_onehot = np.zeros_like(pred)
    for i in range(len(gt)):
        gt_onehot[pred_label[i]] = 1.0
                
    res = OrderedDict({})
    
            
    res['auc'] = roc_auc_score(gt_onehot, pred)
    res['auprc'] = average_precision_score(gt_onehot, pred)
    
    res['\nmat'] = confusion_matrix(gt, pred_label)
    
    for k, v in res.items():
        print(k, ':', v, '|', end='')
    print()
    
    return list(res.values())

if __name__ == '__main__':
    pass
