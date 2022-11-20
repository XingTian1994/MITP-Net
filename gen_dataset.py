# This script generates the dataset from original data
import numpy as np
import pandas as pd
import os
from collections import OrderedDict
from datetime import datetime
import torch


class ROOM:
    def __init__(self, name, da_path='./Dataset_new'):
        self.or_path = './Original_data'
        self.name = name[0:6]
        self.da_path = da_path
        self.csv_name = name + '.csv'

    def read_csv(self):
        files = os.listdir(self.or_path)
        for file in files: 
            if file == self.csv_name:
                self.df = pd.read_csv(self.or_path + '/' + file)
                break
        self.total_cols = self.df.shape[1]
        self.total_rows = self.df.shape[0]
        self.col_list = list(self.df.columns)
    
    def select_t_data(self, date, start_time, end_time):
        start_dati = datetime.strptime(date + start_time, "%Y-%m-%d%H")
        end_dati = datetime.strptime(date + end_time, "%Y-%m-%d%H")
        self.date_data = torch.zeros(600,9)
        i = 0
        for r in range(0, self.total_rows):
            dati = datetime.strptime(self.df.iloc[r,0] + self.df.iloc[r,1][:-3], "%Y/%m/%d%H")
            if dati >= start_dati and dati <= end_dati:
                self.date_data[i,0]=(r % 1440)   # minute of the day
                # if float(self.df.iloc[r,9]) < 1:
                #     fan = 0                          # FCU = off
                # elif float(self.df.iloc[r,9]) >= 1:
                #     fan = float(self.df.iloc[r,11])  # FCU = on and fan = FCU_fan_feedback
                self.date_data[i,1] = float(self.df.iloc[r,9])  # FCU_onoff_feedback
                # if r == 650:
                #     print(self.df.iloc[r])
                for j in range(1,4):
                    if int(self.df.iloc[r,11]) == j:
                        self.date_data[i,j+1] = 1        # FCU fan state[1:4] means the state of fan level 1 to 3
                self.date_data[i,5] = float(self.df.iloc[r,22])  # occupant_num
                self.date_data[i,6] = float(self.df.iloc[r,24])  # temp1
                self.date_data[i,7] = float(self.df.iloc[r,26])  # temp2
                # self.date_data[i].append(float(self.df.iloc[r,28]))  # differential_pressure
                self.date_data[i,8] = float(self.df.iloc[r,29])  # outdoor_temp
                i += 1

    def select_t_data_new(self, date, start_time, end_time):
        start_dati = datetime.strptime(date + start_time, "%Y-%m-%d%H")
        end_dati = datetime.strptime(date + end_time, "%Y-%m-%d%H")
        self.date_data = torch.zeros(600, 6)
        i = 0
        for r in range(0, self.total_rows):
            dati = datetime.strptime(self.df.iloc[r, 0] + self.df.iloc[r, 1][:-3], "%Y/%m/%d%H")
            if dati >= start_dati and dati <= end_dati:
                self.date_data[i, 0] = (r % 1440)  # minute of the day
                #self.date_data[i, 0] = dati
                if float(self.df.iloc[r,9]) < 1:
                    fan = 0                          # FCU = off
                elif float(self.df.iloc[r,9]) >= 1:
                    fan = float(self.df.iloc[r,11])  # FCU = on and fan = FCU_fan_feedback
                self.date_data[i, 1] = fan
                self.date_data[i, 2] = float(self.df.iloc[r, 22])  # occupant_num
                self.date_data[i, 3] = float(self.df.iloc[r, 24])  # temp1
                self.date_data[i, 4] = float(self.df.iloc[r, 26])  # temp2
                # self.date_data[i].append(float(self.df.iloc[r,28]))  # differential_pressure
                self.date_data[i, 5] = float(self.df.iloc[r, 29])  # outdoor_temp
                i += 1
    
    def normalize_t_data(self):
        indoor_t_lower = 22.0
        indoor_t_upper = 32.0
        outdoor_t_lower = 20.0
        outdoor_t_upper = 40.0
        for n in range(0,600):
            self.date_data[n,5] = self.date_data[n,5] / 10
            self.date_data[n,6] = (self.date_data[n,6] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            self.date_data[n,7] = (self.date_data[n,7] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            self.date_data[n,8] = (self.date_data[n,8] - outdoor_t_lower) / (outdoor_t_upper - outdoor_t_lower)

    def normalize_t_data_new(self):
        indoor_t_lower = 22.0
        indoor_t_upper = 32.0
        outdoor_t_lower = 20.0
        outdoor_t_upper = 40.0
        for n in range(0,600):
            self.date_data[n,1] = self.date_data[n,1] / 3
            self.date_data[n,2] = self.date_data[n,2] / 10
            self.date_data[n,3] = (self.date_data[n,3] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            self.date_data[n,4] = (self.date_data[n,4] - indoor_t_lower) / (indoor_t_upper - indoor_t_lower)
            self.date_data[n,5] = (self.date_data[n,5] - outdoor_t_lower) / (outdoor_t_upper - outdoor_t_lower)

    def select_t_data_csv(self, date, start_time, end_time):
        start_dati = datetime.strptime(date + start_time, "%Y-%m-%d%H")
        end_dati = datetime.strptime(date + end_time, "%Y-%m-%d%H")
        col_names = ['date', 'FCU', 'Occupancy', 'Indoor temp1', 'Indoor temp2', 'Outdoor temp']
        self.date_data = pd.DataFrame(index=np.linspace(1,600,600), columns=col_names)
        i = 0
        for r in range(0, self.total_rows):
            dati = datetime.strptime(self.df.iloc[r, 0] + self.df.iloc[r, 1][:-3], "%Y/%m/%d%H")
            if dati >= start_dati and dati <= end_dati:
                self.date_data.iloc[i, 0] = self.df.iloc[r, 0] + ' ' + self.df.iloc[r, 1]
                if float(self.df.iloc[r,9]) < 1:
                    fan = 0                          # FCU = off
                elif float(self.df.iloc[r,9]) >= 1:
                    fan = float(self.df.iloc[r,11])  # FCU = on and fan = FCU_fan_feedback
                self.date_data.iloc[i, 1] = fan
                self.date_data.iloc[i, 2] = float(self.df.iloc[r, 22])  # occupant_num
                self.date_data.iloc[i, 3] = float(self.df.iloc[r, 24])  # temp1
                self.date_data.iloc[i, 4] = float(self.df.iloc[r, 26])  # temp2
                # self.date_data[i].append(float(self.df.iloc[r,28]))  # differential_pressure
                self.date_data.iloc[i, 5] = float(self.df.iloc[r, 29])  # outdoor_temp
                i += 1

    def save_t_data(self, date):
        if not os.path.exists(self.da_path + '/' + date):
            os.makedirs(self.da_path + '/' + date)
        self.date_data.to_csv(self.da_path + '/' + date + '/' + self.name + '.csv', index=None)
    
    def save_tensor_data(self, date):
        #self.date_data -> tensor 
        self.date_data=torch.Tensor(self.date_data)
        for i in range(0, 600):
            if torch.isnan(self.date_data[i,5]):
                print(date, i)
        if not os.path.exists(self.da_path + '/' + date):
            os.makedirs(self.da_path + '/' + date)
        torch.save(self.date_data, self.da_path + '/' + date + '/' + self.name + '.pt')

    

if __name__ == "__main__":
    date_list = ['2021-08-09', '2021-08-10', '2021-08-11', '2021-08-12', '2021-08-13', '2021-08-14', \
                    '2021-08-16', '2021-08-17', '2021-08-18', '2021-08-19', '2021-08-20', '2021-08-21']
    room1 = ROOM("room_1_result", da_path='./Dataset_csv')
    room2 = ROOM("room_2_result", da_path='./Dataset_csv')
    #room3 = ROOM("room_3_result")
    room4 = ROOM("room_4_result", da_path='./Dataset_csv')
    room5 = ROOM("room_5_result", da_path='./Dataset_csv')
    room6 = ROOM("room_6_result", da_path='./Dataset_csv')
    room7 = ROOM("room_7_result", da_path='./Dataset_csv')
    room_list = [room1, room2, room4, room5, room6, room7]
    start = '09'
    end = '18'

    for r in room_list:
        r.read_csv()
        for d in date_list:
            print(r, d)
            r.select_t_data_csv(d, start, end)
            #r.normalize_t_data_new()
            r.save_t_data(d)
    
    
