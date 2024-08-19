# todo:
    # find the right model; 
    # develop stopping procedure to stop training (when does overfitting start?)
    # estimate model on full data? (when we have stopping procedure we need validation data)
    # save into csv: DV, treatment variable, model predictions
    # use R: run regression

from math import floor
import pandas as pd
import torch
import numpy
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") # select first gpu
print("device = " + str(device))
torch.set_printoptions(sci_mode=False, edgeitems=5)
# torch.manual_seed(0)
# torch.cuda.manual_seed_all(0)

df = pd.read_csv("clean/cleanData.csv")

inputs = torch.tensor(df.drop(["score1","choice2"],axis=1).to_numpy(),dtype=torch.float).to(device)
targets = torch.tensor(df["choice2"],dtype=torch.float).to(device)
regressor = torch.tensor(df["score1"],dtype=torch.float).to(device) 

indices = torch.randperm(len(df))   # randomly reorder indices
nfolds = inputs.size()[0]
foldsize = floor(inputs.size()[0]/nfolds)
folds = []
for k in range(nfolds+1):
    folds.append(k*foldsize)
folds[-1] = inputs.size()[0]

full_data = (inputs,targets)

        # Fully connected (dense) neural network
        # size of layer0 = input features ("number of neurons" in input layer)
n1 = 500  # size of output of layer1
n2 = 500  # size of output of layer2
n3 = 500  # size of output of layer3
        # size of output layer = 1 (output of neural network)

class Network(torch.nn.Module):
    def __init__(self):
        super(Network,self).__init__()
        self.linear1 = torch.nn.Linear(inputs.size()[1],n1)
        self.activation = torch.nn.ReLU()
        self.dropout = torch.nn.Dropout(0.5)
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,n3)
        self.linear4 = torch.nn.Linear(n3,1)
    def forward(self,x): # neural network function (take input to create predictions)
        x = self.linear1(x)        
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        return torch.squeeze(x)

models = [Network().to(device) for i in range(nfolds+1)]
optimizers = [torch.optim.Adam(models[i].parameters(), lr=0.001) 
              for i in range(nfolds+1)]
loss_function = torch.nn.MSELoss()
training_losses = []
validation_losses = []
state_dict = models[0].state_dict()

def get_training_data(i):
    if i==0:
        return full_data
    else:
        start_index = folds[i-1]
        end_index = folds[i]
        # training_indices = torch.cat((indices[:start_index],indices[end_index:]),0)
        training_indices = torch.cat((indices[:0],indices[15:]),0)
        training_data = (inputs[training_indices, :],targets[training_indices])
        return training_data

def get_validation_data(i):
    if i==0:
        return full_data
    else:
        start_index = folds[i-1]
        end_index = folds[i]
        # validation_indices = indices[start_index:end_index]
        validation_indices = indices[0:15]
        validation_data = (inputs[validation_indices, :],targets[validation_indices])
        return validation_data

def get_model_loss(data,model):
    model.eval()
    with torch.no_grad():
        x,y = data
        outputs = model(x)
        loss = loss_function(outputs,y)
        return loss.item()

def get_training_loss():
    training_losses = [0 for i in range(nfolds)]
    for k in range(nfolds):
        training_data = get_training_data(k)
        training_losses[k] = get_model_loss(training_data,models[k+1])
    return sum(training_losses)/len(training_losses)
    
def get_validation_loss():
    validation_losses = [0 for i in range(nfolds)]
    for k in range(nfolds):
        validation_data = get_validation_data(k)
        validation_losses[k] = get_model_loss(validation_data,models[k+1])
    return sum(validation_losses)/len(validation_losses)

def model_training_step(data,model,optimizer):
    model.train(True)                   # go into training mode
    optimizer.zero_grad()               # clear out old gradients
    x,y = data                          # y - targets, x - labels
    outputs = model(x)
    loss = loss_function(outputs,y)
    loss.backward()                    # backpropagation
    optimizer.step()
    return loss.item()

def step():
    for k in range(nfolds+1):
        training_data = get_training_data(k)
        model_training_step(training_data,models[k],optimizers[k])

def save_output():
    models[0].load_state_dict(state_dict)
    prediction = models[0](inputs)
    results = {
        "choice2": targets.cpu().detach().tolist(),
        "score1": regressor.cpu().detach().tolist(),
        "prediction": prediction.cpu().detach().tolist(),
        "choice1": df["choice1"].tolist()
    }
    pd.DataFrame(data=results).to_csv("./machineLearning.csv",index=False)
    loss = {
        "trainingLoss": training_losses,
        "validationLoss": validation_losses
    }
    pd.DataFrame(data=loss).to_csv("./loss.csv",index=False)
    print("CSV Written")


nsteps = 500
min_validation_loss = 1000000

for i in range(nsteps):
    step()
    validation_loss = get_validation_loss()
    validation_losses.append(validation_loss)
    training_loss = get_training_loss()
    training_losses.append(training_loss)
    if(validation_loss < min_validation_loss):
        min_validation_loss = validation_loss
        state_dict = models[0].state_dict()
    print("Step",i,", V:",round(validation_loss,3),", T:",round(training_loss,3))    


save_output()
# 5k runs for noisy 100k bootstrapped data
# 3k runs for 23 observations


