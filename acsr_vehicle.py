import torch
from torch.utils.data import Dataset
import torch.nn as nn
import pathlib
import os
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import joblib

class VehicleDataset(Dataset):
    def __init__(self, directory, file_name_init, input_row=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9], output_row=[3, 4], mode=0,
                input_timesteps=5, scaler_mode=0, shuffle=True,scaler_file=None):
        self.mode = mode
        super().__init__()
        #scaler mode, 0: no scaler, 1:scaler, and save the scaler, 2: scaler, but the scalers are loaded from file
        self.scaler_mode = scaler_mode

        file_counting = 0
        parent_dir = pathlib.Path('__file__').parent.resolve()
        trainingdata_dir = os.path.join(parent_dir, directory)

        for file in os.listdir(trainingdata_dir):
            if file.startswith(file_name_init):
                file_counting += 1

        data = [None] * file_counting
        data_length = [0] * file_counting
        for i in range(0, file_counting):
            with open(os.path.join(trainingdata_dir, 'data_to_train') + '_' + str(i) + '.csv', 'r') as fh:
                data[i] = np.loadtxt(fh, delimiter=',')
                data_length[i] = len(data[i])

        input_shape = len(input_row)
        output_shape = len(output_row)

        _,cols = data[0].shape

        self.scalers = [None] * cols

        if scaler_mode == 1:
            total_data = np.concatenate(data, axis=0)
            for i in range(cols):
                scaler = StandardScaler()
                self.scalers[i] = scaler.fit(total_data[:, i])
                for j in range(file_counting):
                    data[j][:, i] = self.scalers[i].transform(data[j][:, i])

            joblib.dump(self.scalers, scaler_file)
        elif scaler_mode==2:
            self.scalers = joblib.load(scaler_file)
            for i in range(cols):
                for j in range(file_counting):
                    data[j][:, i] = self.scalers[i].transform(data[j][:, i])



        total_length = np.sum(data_length) - input_timesteps * file_counting
        data_train = np.zeros((total_length * input_timesteps, input_shape))
        data_labels = np.zeros((total_length, output_shape))

        current_idx = 0
        for u in range(0, file_counting):
            data_labels[current_idx:current_idx + len(data[u]) - input_timesteps] = (data[u])[input_timesteps:,output_row]

            for pp in range(0, len(data[u]) - input_timesteps):
                idx = input_timesteps * current_idx + pp * input_timesteps
                data_train[idx:idx + input_timesteps, :] = (data[u])[pp:pp + input_timesteps, input_row]

            current_idx += (len(data[u]) - input_timesteps)

        if mode == 0:
            data_train = np.reshape(data_train, (len(data_labels), input_timesteps * input_shape))
        elif mode == 1:
            data_train = np.reshape(data_train, (len(data_labels), input_timesteps, input_shape))

        self.x_data = torch.Tensor(data_train)
        self.y_data = torch.Tensor(data_labels)

    def __len__(self):
        return len(self.y_data)

    def __getitem__(self, idx):
        preds = self.x_data[idx, :]
        price = self.y_data[idx, :]
        return (preds, price)

class VehicleFeedForwardModel(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        layers = [nn.Linear(in_c,256),nn.Dropout(), nn.Linear(256,512),nn.Dropout(), nn.Linear(512,out_c)]
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        x = self.layers(x)
        return x