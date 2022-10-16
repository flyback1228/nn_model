from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch.nn as nn
from torch.optim import SGD,Adam

import torch
from tqdm import tqdm
from torch.utils.data import TensorDataset, DataLoader
import pathlib
import os
import numpy as np
from joblib import dump,load

import pathlib




class VehicleForwardFeedNNModel(nn.Module):
    def __init__(self,in_c,out_c):
        super().__init__()
        layers = [nn.Linear(in_c,256),nn.Dropout(), nn.Linear(256,512),nn.Dropout(), nn.Linear(512,out_c)]
        self.layers = nn.Sequential(*layers)

    def forward(self,x):
        x = self.layers(x)
        return x


file_counting = 0
parent_dir = pathlib.Path('__file__').parent.resolve()
trainingdata_dir = os.path.join(parent_dir, "inputs/trainingdata")


for file in os.listdir(trainingdata_dir):
    if file.startswith('data_to_train'):
        file_counting += 1

print('total training filecount: {}'.format(file_counting))


data = [None] * file_counting
data_length = [0] * file_counting

for i in range(0, file_counting):
    with open(os.path.join(trainingdata_dir, 'data_to_train') + '_' + str(i) + '.csv', 'r') as fh:
        data[i] = np.loadtxt(fh, delimiter=',')
        data_length[i] = len(data[i])

print('Load Data Finished')
print('Total Data Set: {}'.format(np.sum(data_length)))


input_shape = 10
output_shape = 2
input_timesteps = 5
ratio = 0.25

total_length = np.sum(data_length) - input_timesteps*file_counting

data_train = np.zeros((total_length * input_timesteps, input_shape))
data_labels = np.zeros((total_length, output_shape))

current_idx = 0

for u in range(0, file_counting):
    data_labels[current_idx:current_idx + len(data[u]) - input_timesteps] = (data[u])[input_timesteps:, 3:3+output_shape]

    for pp in range(0, len(data[u]) - input_timesteps):
        idx = input_timesteps * current_idx + pp * input_timesteps
        data_train[idx:idx + input_timesteps, :] = (data[u])[pp:pp + input_timesteps, :]

    current_idx += ((len(data[u]) - input_timesteps))


data_train = np.reshape(data_train, (len(data_labels), input_timesteps * input_shape))
indices = np.arange(data_train.shape[0])
np.random.shuffle(indices)
data_labels = data_labels[indices]
data_train = data_train[indices]
#data_train = np.reshape(data_train, (len(data_labels) * input_timesteps, input_shape))

#split data
p = int(len(data_train) * (1 - ratio))
mod = p % 5
p = p - mod

train_x = data_train[:p, :]
train_y = data_labels[:p,:]
#x_scalers = StandardScaler()  # with_mean=True, with_std=True
#x_scalers = x_scalers.fit(data_train[:p, :])
#train_x = x_scalers.transform(data_train[:p, :])


#temp = np.zeros((len(data_labels), input_shape))
#temp[:, 0:output_shape] = data_labels
#temp = x_scalers.transform(temp[:p, :])
#train_y = temp[0:(p // input_timesteps), 0:output_shape]
#val_y = temp[(p // input_timesteps):, 0:output_shape]

#train_x = np.reshape(train_x, (p // input_timesteps, input_timesteps * input_shape))
val_x = (data_train[p:, :])
val_y = data_labels[(p // input_timesteps):]
#val_x = x_scalers.transform(data_train[p:, :])
#val_x = np.reshape(val_x, ((len(data_train) - p * input_timesteps), input_timesteps * input_shape))

#train_y = y_scalers.transform(data_train[p:,:])

#shuffle data




print('Traning Data')
print(train_x.shape)
print(train_y.shape)
print(train_x[0:3, :])
print(train_y[0:3, :])

#scale data


train_x = torch.Tensor(train_x) # transform to torch tensor
train_y = torch.Tensor(train_y)
test_x = torch.Tensor(val_x) # transform to torch tensor
test_y = torch.Tensor(val_y)

print("Train X:")
print(train_x[0:2,:])
print("Train Y:")
print(train_y[0:2,:])

train_dataset = TensorDataset(train_x,train_y) # create your datset
train_dataloader = DataLoader(train_dataset,batch_size=10240) # create your dataloader

print('CREATE FEEDFORWARD NEURAL NETWORK')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VehicleForwardFeedNNModel(input_timesteps*input_shape,output_shape).to(device)

criterion = MSELoss().to(device)
optimizer = torch.optim.Adam(model.parameters(),1e-4)

#optimizer = Adam(model.parameters())
#criterion = nn.CrossEntropyLoss()

# enumerate epochs
for epoch in range(500):
    # enumerate mini batches
    with tqdm(train_dataloader, unit="batch") as tepoch:
        for (inputs, targets) in tepoch:
            tepoch.set_description(f"Epoch {epoch}")

            inputs, targets = inputs.to(device), targets.to(device)
            # clear the gradients
            optimizer.zero_grad()
            # compute the model output
            yhat = model(inputs)
            # calculate loss
            loss = criterion(yhat, targets)
            # credit assignment
            loss.backward()
            # update model weights
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())

model_path = os.path.join(parent_dir, "output/scalers");
#dump(x_scalers,model_path)
model_path = os.path.join(parent_dir, "output/trained_model");
torch.save(model, model_path)