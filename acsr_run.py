import os.path

import torch
from tqdm import tqdm
from acsr_vehicle import *
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.nn import MSELoss
import matplotlib.pyplot as plt

def train_model(epochs,model_name):
    #dataset and dataload
    train_dataset = VehicleDataset('inputs/trainingdata','data_to_train',input_timesteps=2)
    train_dataloader = DataLoader(train_dataset, batch_size=2048,shuffle=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    #nn model
    model = VehicleFeedForwardModel(10*2,2).to(device)

    #optimizer and criterion
    criterion = MSELoss().to(device)
    optimizer = Adam(model.parameters(), 1e-3)

    for epoch in range(epochs):
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

    parent_dir = pathlib.Path('__file__').parent.resolve()
    model_path = os.path.join(parent_dir, 'output/models',model_name)
    torch.save(model.state_dict(), model_path)
    return model

def run_model(model):

    test_dataset = VehicleDataset('inputs/trainingdata', 'data_to_run',input_timesteps=2)
    test_dataloader = DataLoader(test_dataset,shuffle=False)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    #model = model.to(device)

    criterion = MSELoss()
    losses = []
    predicted_data = torch.empty((len(test_dataset),2))

    for i,(input, gt) in enumerate(test_dataloader):
        #input = input.to(device)
        #gt = gt.to(device)
        predicted = model(input)
        loss = criterion(predicted, gt)
        losses.append(loss)
        predicted_data[i,:] = predicted

    return test_dataset.y_data, predicted_data

if __name__=='__main__':
    model = train_model(400, 'trained')

    #parent_dir = pathlib.Path('__file__').parent.resolve()
    #trainingdata_dir = os.path.join(parent_dir, model)

    model_name = 'output/models/trained'
    if not os.path.exists(model_name):
        print('No trained model found')
        exit
    model = VehicleFeedForwardModel(10*2, 2)
    model.load_state_dict(torch.load(model_name))

    targets,predicted = run_model(model)
    targets=targets.detach().numpy()
    predicted = predicted.detach().numpy()
    plt.plot(targets[0:40,0],'--')
    plt.plot(predicted[0:40,0])
    plt.show()

