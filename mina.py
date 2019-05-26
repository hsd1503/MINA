from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pack_padded_sequence
from scipy.signal import periodogram

import numpy as np
import dill
from collections import Counter
from time import localtime, strftime
import random

import matplotlib.pyplot as plt
import random
from util import evaluate
from util import slide_and_cut
from util import make_challenge_data
import sys, os
from shutil import copyfile

class Net(nn.Module):
    def __init__(self, n_channel, n_dim, n_split):
        super(Net, self).__init__()
        
        self.n_channel = n_channel
        self.n_dim = n_dim
        self.n_split = n_split
        self.n_class = 4
        
        self.base_net_0 = BaseNet(self.n_dim, self.n_split)
        self.base_net_1 = BaseNet(self.n_dim, self.n_split)
        self.base_net_2 = BaseNet(self.n_dim, self.n_split)
        self.base_net_3 = BaseNet(self.n_dim, self.n_split)
            
        ### attention
        self.out_size = 8
        self.att_channel_dim = 2
        self.W_att_channel = nn.Parameter(torch.randn(self.out_size+1, self.att_channel_dim))
        self.v_att_channel = nn.Parameter(torch.randn(self.att_channel_dim, 1))
        
        ### fc
        self.fc = nn.Linear(self.out_size, self.n_class)
        
    def forward(self, x_0, x_1, x_2, x_3, 
                k_beat_0, k_beat_1, k_beat_2, k_beat_3, 
                k_rhythm_0, k_rhythm_1, k_rhythm_2, k_rhythm_3, 
                k_freq):

        x_0, alpha_0, beta_0 = self.base_net_0(x_0, k_beat_0, k_rhythm_0)
        x_1, alpha_1, beta_1 = self.base_net_1(x_1, k_beat_1, k_rhythm_1)
        x_2, alpha_2, beta_2 = self.base_net_2(x_2, k_beat_2, k_rhythm_2)
        x_3, alpha_3, beta_3 = self.base_net_3(x_3, k_beat_3, k_rhythm_3)
        
        x = torch.stack([x_0, x_1, x_2, x_3], 1)

        # ############################################
        # ### attention on channel
        # ############################################
        k_freq = k_freq.permute(1, 0, 2)

        tmp_x = torch.cat((x, k_freq), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_channel)
        e = torch.matmul(torch.tanh(e), self.v_att_channel)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        gama = torch.div(n1, n2)
        x = torch.sum(torch.mul(gama, x), 1)
        
        ############################################
        ### fc
        ############################################
        x = F.softmax(self.fc(x), 1)
        
        ############################################
        ### return 
        ############################################
        
        att_dic = {"alpha_0":alpha_0, "beta_0":beta_0, 
                  "alpha_1":alpha_1, "beta_1":beta_1, 
                  "alpha_2":alpha_2, "beta_2":beta_2, 
                  "alpha_3":alpha_3, "beta_3":beta_3, 
                  "gama":gama}
        
        return x, att_dic

class BaseNet(nn.Module):
    def __init__(self, n_dim, n_split):
        super(BaseNet, self).__init__()
        
        self.n_dim = n_dim
        self.n_split = n_split
        self.n_seg = int(n_dim/n_split)
        
        ### Input: (batch size, number of channels, length of signal sequence)
        self.conv_out_channels = 64
        self.conv_kernel_size = 32
        self.conv_stride = 2
        self.conv = nn.Conv1d(in_channels=1, 
                              out_channels=self.conv_out_channels, 
                              kernel_size=self.conv_kernel_size, 
                              stride=self.conv_stride)
        self.conv_k = nn.Conv1d(in_channels=1, 
                                out_channels=1, 
                                kernel_size=self.conv_kernel_size, 
                                stride=self.conv_stride)
        self.att_cnn_dim = 8
        self.W_att_cnn = nn.Parameter(torch.randn(self.conv_out_channels+1, self.att_cnn_dim))
        self.v_att_cnn = nn.Parameter(torch.randn(self.att_cnn_dim, 1))
        
        ### Input: (batch size, length of signal sequence, input_size)
        self.rnn_hidden_size = 32
        self.lstm = nn.LSTM(input_size=(self.conv_out_channels), 
                            hidden_size=self.rnn_hidden_size, 
                            num_layers=1, batch_first=True, bidirectional=True)
        self.att_rnn_dim = 8
        self.W_att_rnn = nn.Parameter(torch.randn(2*self.rnn_hidden_size+1, self.att_rnn_dim))
        self.v_att_rnn = nn.Parameter(torch.randn(self.att_rnn_dim, 1))
        
        ### fc
        self.do = nn.Dropout(p=0.5)
        self.out_size = 8
        self.fc = nn.Linear(2*self.rnn_hidden_size, self.out_size)
    
    def forward(self, x, k_beat, k_rhythm):
        
        self.batch_size = x.size()[0]

        ############################################
        ### reshape
        ############################################
        # print('orignial x:', x.size())
        x = x.view(-1, self.n_split)
        x = x.unsqueeze(1)
        
        k_beat = k_beat.view(-1, self.n_split)
        k_beat = k_beat.unsqueeze(1)
        
        ############################################
        ### conv
        ############################################
        x = F.relu(self.conv(x))
        
        k_beat = F.relu(self.conv_k(k_beat))
        
        ############################################
        ### attention conv
        ############################################
        x = x.permute(0, 2, 1)
        k_beat = k_beat.permute(0, 2, 1)
        tmp_x = torch.cat((x, k_beat), dim=-1)
        e = torch.matmul(tmp_x, self.W_att_cnn)
        e = torch.matmul(torch.tanh(e), self.v_att_cnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        alpha = torch.div(n1, n2)
        x = torch.sum(torch.mul(alpha, x), 1)
        
        ############################################
        ### reshape for rnn
        ############################################
        x = x.view(self.batch_size, self.n_seg, -1)
    
        ############################################
        ### rnn        
        ############################################
        
        k_rhythm = k_rhythm.unsqueeze(-1)
        o, (ht, ct) = self.lstm(x)
        tmp_o = torch.cat((o, k_rhythm), dim=-1)
        e = torch.matmul(tmp_o, self.W_att_rnn)
        e = torch.matmul(torch.tanh(e), self.v_att_rnn)
        n1 = torch.exp(e)
        n2 = torch.sum(torch.exp(e), 1, keepdim=True)
        beta = torch.div(n1, n2)
        x = torch.sum(torch.mul(beta, o), 1)
        
        ############################################
        ### fc
        ############################################
        x = F.relu(self.fc(x))
        x = self.do(x)
        
        return x, alpha, beta        


def train(model, optimizer, loss_func, epoch, batch_size, 
          X_train, Y_train, K_train_beat, K_train_rhythm, K_train_freq, 
          log_file):
    """
    X_train: (n_channel, n_sample, n_dim)
    Y_train: (n_sample,)
    
    K_train_beat: (n_channel, n_sample, n_dim)
    K_train_rhythm: (n_channel, n_sample, n_dim/n_split)
    K_train_freq: (n_channel, n_sample)
    """
    model.train()
    
    n_train = len(Y_train)
    
    pred_all = []
    batch_start_idx = 0
    batch_end_idx = 0
    loss_all = []
    while batch_end_idx < n_train:
        print('.', end="")
        batch_end_idx = batch_end_idx + batch_size
        if batch_end_idx >= n_train:
            batch_end_idx = n_train
            
        ### input data
        batch_input_0 = Variable(torch.FloatTensor(X_train[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_1 = Variable(torch.FloatTensor(X_train[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_2 = Variable(torch.FloatTensor(X_train[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_3 = Variable(torch.FloatTensor(X_train[3, batch_start_idx: batch_end_idx, :])).cuda()
        
        ### input K_beat
        batch_K_beat_0 = Variable(torch.FloatTensor(K_train_beat[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_1 = Variable(torch.FloatTensor(K_train_beat[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_2 = Variable(torch.FloatTensor(K_train_beat[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_3 = Variable(torch.FloatTensor(K_train_beat[3, batch_start_idx: batch_end_idx, :])).cuda()

        ### input K_rhythm
        batch_K_rhythm_0 = Variable(torch.FloatTensor(K_train_rhythm[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_1 = Variable(torch.FloatTensor(K_train_rhythm[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_2 = Variable(torch.FloatTensor(K_train_rhythm[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_3 = Variable(torch.FloatTensor(K_train_rhythm[3, batch_start_idx: batch_end_idx, :])).cuda()        
        
        ### input K_freq
        batch_K_freq = Variable(torch.FloatTensor(K_train_freq[:, batch_start_idx: batch_end_idx, :])).cuda()  
        
        ### gt
        batch_gt = Variable(torch.LongTensor(Y_train[batch_start_idx: batch_end_idx])).cuda()
        
        pred, _ = model(batch_input_0, batch_input_1, batch_input_2, batch_input_3, 
                        batch_K_beat_0, batch_K_beat_1, batch_K_beat_2, batch_K_beat_3, 
                        batch_K_rhythm_0, batch_K_rhythm_1, batch_K_rhythm_2, batch_K_rhythm_3, 
                        batch_K_freq)
        
        pred_all.append(pred.cpu().data.numpy())
        # print(pred, batch_gt)

        loss = loss_func(pred, batch_gt)
        loss_all.append(loss.cpu().data.numpy())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_start_idx = batch_start_idx + batch_size

    loss_res = np.mean(loss_all)
    print('epoch {0} '.format(epoch))
    print('loss ', np.mean(loss_all))
    print('train | ', end='')
    pred_all = np.concatenate(pred_all, axis=0)
    # print(Y_train.shape, pred_all.shape)
    res = evaluate(Y_train, pred_all)
    res.append(loss_res)
    res.append(pred_all)
    
    with open(log_file, 'a') as fout:
        print('epoch {0} '.format(epoch), 'train | ', res, file=fout)
        print('loss_all ', np.mean(loss_all), file=fout)
        
    return res
    

def test(model, batch_size, 
         X_test, Y_test, K_test_beat, K_test_rhythm, K_test_freq, 
         log_file):
    
    model.eval()
    
    n_test = len(Y_test)
    
    pred_all = []
    att_dic_all = []
    
    batch_start_idx = 0
    batch_end_idx = 0
    while batch_end_idx < n_test:
        print('.', end="")
        batch_end_idx = batch_end_idx + batch_size
        if batch_end_idx >= n_test:
            batch_end_idx = n_test
            
        ### input data
        batch_input_0 = Variable(torch.FloatTensor(X_test[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_1 = Variable(torch.FloatTensor(X_test[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_2 = Variable(torch.FloatTensor(X_test[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_input_3 = Variable(torch.FloatTensor(X_test[3, batch_start_idx: batch_end_idx, :])).cuda()
        
        ### input K_beat
        batch_K_beat_0 = Variable(torch.FloatTensor(K_test_beat[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_1 = Variable(torch.FloatTensor(K_test_beat[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_2 = Variable(torch.FloatTensor(K_test_beat[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_beat_3 = Variable(torch.FloatTensor(K_test_beat[3, batch_start_idx: batch_end_idx, :])).cuda()

        ### input K_rhythm
        batch_K_rhythm_0 = Variable(torch.FloatTensor(K_test_rhythm[0, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_1 = Variable(torch.FloatTensor(K_test_rhythm[1, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_2 = Variable(torch.FloatTensor(K_test_rhythm[2, batch_start_idx: batch_end_idx, :])).cuda()
        batch_K_rhythm_3 = Variable(torch.FloatTensor(K_test_rhythm[3, batch_start_idx: batch_end_idx, :])).cuda()
        
        ### input K_freq
        batch_K_freq = Variable(torch.FloatTensor(K_test_freq[:, batch_start_idx: batch_end_idx, :])).cuda()
        
        ### gt
        batch_gt = Variable(torch.LongTensor(Y_test[batch_start_idx: batch_end_idx])).cuda()

        pred, att_dic = model(batch_input_0, batch_input_1, batch_input_2, batch_input_3, 
                              batch_K_beat_0, batch_K_beat_1, batch_K_beat_2, batch_K_beat_3, 
                              batch_K_rhythm_0, batch_K_rhythm_1, batch_K_rhythm_2, batch_K_rhythm_3, 
                              batch_K_freq)
            
        for k, v in att_dic.items():
            att_dic[k] = v.cpu().data.numpy()
        att_dic_all.append(att_dic)
        pred_all.append(pred.cpu().data.numpy())

        batch_start_idx = batch_start_idx + batch_size

    print('test | ', end='')
    pred_all = np.concatenate(pred_all, axis=0)
    res = evaluate(Y_test, pred_all)
    res.append(pred_all)
    
    with open(log_file, 'a') as fout:
        print('test | ', res, file=fout)

    return res, att_dic_all

def compute_beat(X):
    out = np.zeros((X.shape[0], X.shape[1], X.shape[2]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i, j] = np.concatenate([[0], np.abs(np.diff(X[i,j,:]))])
    return out

def compute_rhythm(X, n_split):
    cnt_split = int(X.shape[2]/n_split)
    out = np.zeros((X.shape[0], X.shape[1], cnt_split))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            tmp_ts = X[i,j,:]
            tmp_ts_cut = np.split(tmp_ts, X.shape[2]/n_split)
            for k in range(cnt_split):
                out[i, j, k] = np.std(tmp_ts_cut[k])
    return out

def compute_freq(X):
    out = np.zeros((X.shape[0], X.shape[1], 1))
    fs = 300
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            _, Pxx_den = periodogram(X[i,j,:], fs)
            out[i, j, 0] = np.sum(Pxx_den)
    return out


def run():

    n_epoch = 200
    lr = 0.003
    n_split = 50

    ##################################################################
    ### par
    ##################################################################
    run_id = 'mina_{0}'.format(strftime("%Y-%m-%d-%H-%M-%S", localtime()))
    directory = 'res/{0}'.format(run_id)
    try:
        os.stat(directory)
    except:
        os.mkdir(directory)
    
    log_file = '{0}/log.txt'.format(directory)
    model_file = 'mina.py'
    copyfile(model_file, '{0}/{1}'.format(directory, model_file))

    n_dim = 3000
    batch_size = 128

    with open(log_file, 'a') as fout:
        print(run_id, file=fout)

    ##################################################################
    ### read data
    ##################################################################
    with open('data/label.pkl', 'rb') as fin:
        res = dill.load(fin)    
    Y_train = res['Y_train']
    Y_val = res['Y_val']
    Y_test = res['Y_test']
    fin = open('data/X_train.bin', 'rb')
    X_train = np.load(fin)
    fin.close()
    fin = open('data/X_val.bin', 'rb')
    X_val = np.load(fin)
    fin.close()
    fin = open('data/X_test.bin', 'rb')
    X_test = np.load(fin)
    fin.close()

    X_train = np.swapaxes(X_train, 0, 1)
    X_val = np.swapaxes(X_val, 0, 1)
    X_test = np.swapaxes(X_test, 0, 1)
    print(X_train.shape, X_val.shape, X_test.shape)
    
    ##################################################################
    ### compute knowledge
    ##################################################################
    K_train_beat = compute_beat(X_train)
    K_train_rhythm = compute_rhythm(X_train, n_split)
    K_train_freq = compute_freq(X_train)

    K_val_beat = compute_beat(X_val)
    K_val_rhythm = compute_rhythm(X_val, n_split)
    K_val_freq = compute_freq(X_val)

    K_test_beat = compute_beat(X_test)
    K_test_rhythm = compute_rhythm(X_test, n_split)
    K_test_freq = compute_freq(X_test)    
    
    ##################################################################
    ### train
    ##################################################################

    n_channel = X_train.shape[0]
    print('n_channel:', n_channel)

    torch.cuda.manual_seed(0)

    model = Net(n_channel, n_dim, n_split)
    model.cuda()

    optimizer = optim.Adam(model.parameters(), lr=lr)
    print(np.sum(Y_train, axis=0))
    # weight = Variable(torch.FloatTensor([n_train/cnter[0], n_train/cnter[1]])).cuda()
    loss_func = torch.nn.CrossEntropyLoss()

    train_res_list = []
    val_res_list = []
    test_res_list = []
    val_att_list = []
    test_att_list = []
    for epoch in range(n_epoch):
        tmp_train = train(model, optimizer, loss_func, epoch, batch_size, 
                          X_train, Y_train, K_train_beat, K_train_rhythm, K_train_freq, 
                          log_file)
        tmp_val, tmp_att_val = test(model, batch_size, 
                                    X_val, Y_val, K_val_beat, K_val_rhythm, K_val_freq, 
                                    log_file)
        tmp_test, tmp_att_test = test(model, batch_size, 
                                      X_test, Y_test, K_test_beat, K_test_rhythm, K_test_freq, 
                                      log_file)
        
        train_res_list.append(tmp_train)
        val_res_list.append(tmp_val)
        test_res_list.append(tmp_test)
        # val_att_list.append(tmp_att_val)
        test_att_list.append(tmp_att_test)
        torch.save(model, '{0}/model_{1}.pt'.format(directory, epoch))
    
    ##################################################################
    ### save results
    ##################################################################
    res_mat = []
    for i in range(n_epoch):
        train_res = train_res_list[i]
        val_res = val_res_list[i]
        test_res = test_res_list[i]
        res_mat.append([
            train_res[0], train_res[1], 
            val_res[0], val_res[1], 
            test_res[0], test_res[1]])
    res_mat = np.array(res_mat)

    res = {'train_res_list':train_res_list, 
           'val_res_list':val_res_list, 
           'test_res_list':test_res_list}
    with open('{0}/res.pkl'.format(directory), 'wb') as fout:
        dill.dump(res, fout)
    

    np.savetxt('{0}/res_mat.csv'.format(directory), res_mat, delimiter=',')
    

    try:
        res = {'test_att_list':test_att_list}
        with open('{0}/res_att.pkl'.format(directory), 'wb') as fout:
            dill.dump(res, fout)
    except:
        print('error in saving attention file')
    
if __name__ == "__main__":

    make_challenge_data()
    
    for i_run in range(10):
        run()



