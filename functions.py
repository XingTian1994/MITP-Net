from refs import *
import torch
import torch.nn as nn
import random
import pandas as pd
import numpy as np
import os


def evaluation(tr_y, real_y, bound):
    sum = 0
    for b in range(len(real_y)):
        for i in range(len(real_y[0])):  # output timewindow
            for j in range(len(real_y[0][0])):  # 6 zones * 2 sensors
                if abs(tr_y[b][i][j] - real_y[b][i][j]) <= bound:
                    sum += 1
    ratio = sum / (len(real_y) * len(real_y[0]) * len(real_y[0][0])) * 100
    return ratio


def evaluation1(tr_y, real_y, bound):
    sum = 0
    for b in range(len(real_y)):
        for j in range(len(real_y[0][0])):  # 6 zones * 2 sensors
            if abs(tr_y[b][-1][j] - real_y[b][-1][j]) <= bound:
                sum += 1
    ratio = sum / (len(real_y) * len(real_y[0][0])) * 100
    return ratio


def seed_torch(seed=1):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # 设置 cpu 的随机数种子
    torch.cuda.manual_seed(seed)  # 对于单张显卡，设置 gpu 的随机数种子
    # torch.cuda.manual_seed_all(seed) # 对于多张显卡，设置所有 gpu 的随机数种子
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def input_encode(data):
    res = torch.zeros(timewindow-1,48)
    for n in range(timewindow-1):
        for i in range(6):
            res[n,8*i+int(data[6*i+1])] = 1
            res[n,8*i+4] = data[n, 6*i+2] / 10
            res[n,8*i+5] = (data[n, 6*i+3] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            res[n,8*i+6] = (data[n, 6*i+4] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            res[n,8*i+7] = (data[n, 6*i+5] - outdoor_t_lower) / (outdoor_t_upper - outdoor_t_lower)
    return res


def input_encode_new(data):
    res = torch.zeros(timewindow-1,30)
    for n in range(timewindow-1):
        for i in range(6):
            res[n,5*i] = data[n, 5*i] / 3
            res[n,5*i+1] = data[n, 5*i+1] / 10
            res[n,5*i+2] = (data[n, 5*i+2] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            res[n,5*i+3] = (data[n, 5*i+3] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            res[n,5*i+4] = (data[n, 5*i+4] - outdoor_t_lower) / (outdoor_t_upper - outdoor_t_lower)
    return res


def output_encode(data):
    res = torch.zeros(30, 12)
    for n in range(len(data)):
        for i in range(6):
            res[n,2*i] = (data[n, 2*i] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            res[n,2*i+1] = (data[n, 2*i+1] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
    return res


def input_decode(data):
    res = torch.zeros(36)
    for d in range(len(data)):
        if d % 8 in [0, 1, 2, 3]:
            res[int(d/8)*6] = data[d]
        elif d % 8 == 4:
            res[int(d/8)*6+2] = data[d] * 10
        elif d % 8 == 5:
            res[int(d/8)*6+3] = data[d]*(indoor_t_upper - indoor_t_lower) + indoor_t_lower
        elif d % 8 == 6:
            res[int(d/8)*6+4] = data[d]*(indoor_t_upper - indoor_t_lower) + indoor_t_lower
        elif d % 8 == 7:
            res[int(d/8)*6+5] = data[d]*(outdoor_t_upper - outdoor_t_lower) + outdoor_t_lower
    return res

def input_decode_to(data):
    res = torch.zeros(12)
    for d in range(len(data)):
       if d%2 == 1:
           res[d] = data[d] * (outdoor_t_upper - outdoor_t_lower) + outdoor_t_lower
    return res


def input_decode_new(data):
    res = torch.zeros(len(data),len(data[0]))
    for r in range(len(data)):
        for d in range(len(data[0])):
            if d % 5 == 0:
                res[r, d] = data[r, d] * 3
            elif d % 5 == 1:
                res[r,d] = data[r,d] * 10
            elif d % 5 == 2:
                res[r,d] = data[r,d]*(indoor_t_upper - indoor_t_lower) + indoor_t_lower
            elif d % 5 == 3:
                res[r,d] = data[r,d]*(indoor_t_upper - indoor_t_lower) + indoor_t_lower
            elif d % 5 == 4:
                res[r,d] = data[r,d]*(outdoor_t_upper - outdoor_t_lower) + outdoor_t_lower
    return res


def output_decode(data):
    res = torch.zeros(len(data), 12)
    for r in range(len(data)):
        for d in range(len(data[0])):
            res[r,d] = data[r,d]*(indoor_t_upper - indoor_t_lower) + indoor_t_lower
    return res


def output_decode2(data):
    res = np.zeros(12)
    for d in range(len(data)):
        res[d] = data[d] * (indoor_t_upper - indoor_t_lower) + indoor_t_lower
    return res

def BATCH_CVRMSE(y_test, y_real):
    tot = 0
    for b in range(len(y_real)):
        total = len(y_real[0]) * len(y_real[0][0])
        s = 0
        y_ = 0
        for i in range(len(y_real)):
            for j in range(len(y_real[0])):
                s += torch.square(y_test[i,j]-y_real[i,j]) / total
                y_ += y_real[i,j] / total
        res = pow(s, 0.5) / y_ * 100
        tot += torch.sum(res)
    return tot