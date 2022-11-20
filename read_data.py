import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

class MyDataset(Dataset):
    def __init__(self, datelist, name, length=1, timewindow=30, mode=1, label_start=1, label_end=30, read_type='ori'):
        self.label_start = label_start
        self.label_end = label_end
        self.read_type = read_type
        if mode == 1:  # single zone dataset
            if read_type == 'ori':
                or_data = torch.empty(1, 8)
            elif read_type == 'new':
                or_data = torch.empty(1, 5)
            labels = torch.empty(1, 2)
            for i in range(len(datelist)):
                if read_type == 'ori':
                    or_dataper = torch.load('./Dataset/' + datelist[i] + '/' + name)
                    or_dataper = or_dataper[:, 1:]
                    labelsper = or_dataper[:, 5:7]
                    or_data = torch.cat((or_data, or_dataper), 0)
                    labels = torch.cat((labels, labelsper), 0)
                elif read_type == 'new':
                    or_dataper = torch.load('./Dataset_new/' + datelist[i] + '/' + name)
                    or_dataper = or_dataper[:, 1:]
                    labelsper = or_dataper[:, 2:4]
                    or_data = torch.cat((or_data, or_dataper), 0)
                    labels = torch.cat((labels, labelsper), 0)

                self.data = or_data[1:, :]
                self.labels = labels[1:, :]
                self.timewindow = timewindow
        elif mode == 2:
            self.data = torch.empty(length, 1)
            self.labels = torch.empty(length, 1)
            self.timewindow = timewindow

    def __getitem__(self, index):
        if (600 - index % 600) > self.label_end:
            data = self.data[index:index + self.timewindow]
            label = self.labels[index + self.label_start:index + self.label_end]
            self.data_his = data
            self.label_his = label
            return data, label
        else:
            return self.data_his, self.label_his

    def __len__(self):
        return len(self.data)
