# interaction of choice 1 and score 1 non-linearly
# https://en.wikipedia.org/wiki/Universal_approximation_theorem

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

inputs = torch.tensor((df[["score1","choice1"]]).to_numpy(),dtype=torch.float).to(device)
demographics = torch.tensor(df.drop(["score1","choice1","choice2","study"],axis=1).to_numpy(),dtype=torch.float).to(device)
targets = torch.tensor(df["choice2"],dtype=torch.float).to(device)

indices = torch.randperm(len(df))   # randomly reorder indices
nfolds = inputs.size()[0]
foldsize = floor(inputs.size()[0]/nfolds)
folds = []
for k in range(nfolds+1):
    folds.append(k*foldsize)
folds[-1] = inputs.size()[0]

full_data = (inputs,demographics,targets)

        # Fully connected (dense) neural network
        # size of layer0 = input features ("number of neurons" in input layer)
n1 = 500  # size of output of layer1
n2 = 500  # size of output of layer2
n3 = 500  # size of output of layer3
        # size of output layer = 1 (output of neural network)

# In ML
# y = mx (linear function)
# y = mx + 1*b (affine function)
# intercept is bias (from linear to affine)
# https://www.pico.net/kb/the-role-of-bias-in-neural-networks/
# 3*500 +  501*500 + 501*500 + 501*1 = 503,001 parameters

class Simple_Network(torch.nn.Module):
    def __init__(self):
        super(Simple_Network,self).__init__()
        self.linear1 = torch.nn.Linear(inputs.size()[1],n1)
        self.activation = torch.nn.ReLU()
        self.sigma = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,n3)
        self.linear4 = torch.nn.Linear(n3,1)
    def forward(self,x,z): # neural network function (take input to create predictions)
        x = self.linear1(x)        
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = 0.5*self.sigma(x)
        return torch.squeeze(x)

class DemoNetwork(torch.nn.Module):
    def __init__(self):
        super(DemoNetwork,self).__init__()
        self.linear1 = torch.nn.Linear(inputs.size()[1]+demographics.size()[1],n1)
        self.activation = torch.nn.ReLU()
        self.sigma = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(n1,n2)
        self.linear3 = torch.nn.Linear(n2,n3)
        self.linear4 = torch.nn.Linear(n3,1)
    def forward(self,x,z):
        x = torch.cat((x,z),1)
        x = self.linear1(x)        
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        x = self.activation(x)
        x = self.linear4(x)
        x = 0.5*self.sigma(x)
        return torch.squeeze(x)

class SeparableDemoNetwork(torch.nn.Module):
    def __init__(self):
        super(SeparableDemoNetwork,self).__init__()
        self.linear1x = torch.nn.Linear(inputs.size()[1],n1)
        self.linear2x = torch.nn.Linear(n1,n2)
        self.linear3x = torch.nn.Linear(n2,n3)
        self.linear4x = torch.nn.Linear(n3,1)
        self.linear1z = torch.nn.Linear(demographics.size()[1],n1)
        self.linear2z = torch.nn.Linear(n1,n2)
        self.linear3z = torch.nn.Linear(n2,n3)
        self.linear4z = torch.nn.Linear(n3,1)
        self.linear5 = torch.nn.Linear(2,1)
        self.activation = torch.nn.ReLU()
        self.sigma = torch.nn.Sigmoid()
    def forward(self,x,z):
        x = self.linear1x(x)        
        x = self.activation(x)
        x = self.linear2x(x)
        x = self.activation(x)
        x = self.linear3x(x)
        x = self.activation(x)
        x = self.linear4x(x)
        z = self.linear1z(z)        
        z = self.activation(z)
        z = self.linear2z(z)
        z = self.activation(z)
        z = self.linear3z(z)
        z = self.activation(z)
        z = self.linear4z(z)
        y = self.linear5(torch.cat((x,z),1))
        y = 0.5*self.sigma(y)
        return torch.squeeze(y)

Network_Class = Simple_Network  # change the network here!
models = [Network_Class().to(device) for i in range(nfolds+1)]
optimizers = [torch.optim.Adam(models[i].parameters(), lr=0.0001) 
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
        training_indices = torch.cat((indices[:start_index],indices[end_index:]),0)
        training_data = (
            inputs[training_indices, :],
            demographics[training_indices, :],
            targets[training_indices]
        )
        return training_data

def get_validation_data(i):
    if i==0:
        return full_data
    else:
        start_index = folds[i-1]
        end_index = folds[i]
        validation_indices = indices[start_index:end_index]
        validation_data = (
            inputs[validation_indices, :],
            demographics[validation_indices, :],
            targets[validation_indices]
        )
        return validation_data

def get_model_loss(data,model):
    model.eval() # no backpropagation
    with torch.no_grad(): # no backpropagation
        x,z,y = data
        outputs = model(x,z)
        loss = loss_function(outputs,y)
        return loss.item()

def get_training_loss(): # no backpropagation
    training_losses = [0 for i in range(nfolds)]
    for k in range(nfolds):
        training_data = get_training_data(k+1)
        training_losses[k] = get_model_loss(training_data,models[k+1])
    return sum(training_losses)/len(training_losses)
    
def get_validation_loss():  # no backpropagation
    validation_losses = [0 for i in range(nfolds)]
    for k in range(nfolds):
        validation_data = get_validation_data(k+1)
        validation_losses[k] = get_model_loss(validation_data,models[k+1])
    return sum(validation_losses)/len(validation_losses)

def model_training_step(data,model,optimizer): # backpropagation
    model.train(True)                   
    optimizer.zero_grad()               
    x,z,y = data
    outputs = model(x,z)
    loss = loss_function(outputs,y)
    loss.backward() # backpropagation
    optimizer.step()
    return loss.item()

def step():
    for k in range(nfolds+1):
        training_data = get_training_data(k)
        model_training_step(training_data,models[k],optimizers[k])

def save_output():
    print("Save Output")
    models[0].load_state_dict(state_dict)
    prediction = models[0](inputs,demographics)
    results = {
        "choice2":  df["choice2"].tolist(),
        "prediction": prediction.cpu().detach().tolist(),
        "choice1": df["choice1"].tolist(),
        "score1": df["score1"].tolist()
    }
    pd.DataFrame(data=results).to_csv("./machineLearning.csv",index=False)
    print("Machine learning CSV Written")
    loss = {
        "trainingLoss": training_losses,
        "validationLoss": validation_losses
    }
    pd.DataFrame(data=loss).to_csv("./loss.csv",index=False)
    print("Loss CSV Written")
    steps = [i*0.01 for i in range(0,51)]
    grid = {
        choice1: [predict(score1,choice1) for score1 in steps] 
        for choice1 in steps
    }
    grid_data_frame = pd.DataFrame(grid)
    grid_data_frame.index = steps
    grid_data_frame.to_csv('grid.csv')
    print("Grid CSV Written")

def predict(score1,choice1):
    print("Predict",score1,choice1)
    x = torch.tensor([[score1,choice1]]).to(device)
    if(Network_Class == Simple_Network):
        return models[0](x,demographics[0]).cpu().detach().item()
    predictions = [0 for i in range(inputs.size()[0])]
    for i in range(inputs.size()[0]):
        predictions[i] = models[0](x,demographics[i]).cpu().detach().item()
    return sum(predictions)/len(predictions)



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
    print("Step",i,", V:",round(validation_loss,5),", T:",round(training_loss,5))    

save_output()