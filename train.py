import json
from model import NeuralNet

from file import tokenize,stem,bow
import numpy as np
import torch
import torch.nn as nn # nn is a module for Neural Networks
from torch.utils.data import Dataset,DataLoader

with open('data.json','r') as f:
    data=json.load(f)

all_words=[] # we need all words 
tags=[]
xy=[] # this will hold both of our patterns
for d in data['intents']:
    tag=d['tag']
    tags.append(tag) # we are collecting all tags

    for pattern in d['patterns']:
        w=tokenize(pattern)  # Tokeinizing the patterns
        all_words.extend(w)  # Adding it to all_words
        xy.append((w,tag))   # Here we are collecting our training data

ignore=['?','!',',',"'s",'.']

all_words=[stem(w)for w in all_words if w not in ignore] 


tags=sorted(set(tags))
print(tags)

#Creating Data
X_train=[] 
y_train=[]

for (pattern_sentence,tag) in xy:
    bag=bow(pattern_sentence,all_words) 
    X_train.append(bag)

    label=tags.index(tag)
    y_train.append(label) 

X_train=np.array(X_train)
y_train=np.array(y_train)


# Creating PyTorch DataSet

class ChatData(Dataset):
    def __init__(self):
        self.n_samples=len(X_train)
        self.x_data=X_train
        self.y_data=y_train
    
    def __getitem__(self,index):
        return self.x_data[index],self.y_data[index]
    
    def __len__(self):
        return self.n_samples



#HyperParameters
b_size=8 
hidden_size=8
output_size=len(tags) # Number of different classes we have
input_size=len(X_train[0]) 

learning_rate=0.001 
num_epocs=1000 #  number of complete passes through the training dataset.


print(input_size,len(all_words))
print(output_size,tags)
Dataset=ChatData()
training_loader=DataLoader(dataset=Dataset,batch_size=b_size,shuffle=True,num_workers=0)
# num_workers:how many parallel subprocesses you want to activate when you are loading
# all your data during your training or validation.

# we can check if we have GPU support 
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=NeuralNet(input_size,hidden_size,output_size).to(device) 
# Pushing the model to our device 

criterion=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(model.parameters(),lr=learning_rate)

for epoch in range(num_epocs):
    for(words,labels) in training_loader:
        words=words.to(device)
        labels=labels.to(device,dtype=torch.long)
        outputs=model(words)
        loss=criterion(outputs,labels)
        optimizer.zero_grad() 
        loss.backward()
        optimizer.step()

    if(epoch+1)%100==0:
        print(f'Epoch {(epoch+1)/num_epocs}, loss= {loss.item():.4f}')


print(f'Final Loss: ,loss= {loss.item():.4f}')

# Save Data
data={
    "model_state":model.state_dict(),
    "input_size":input_size,
    "output_size":output_size,
    "hidden_size":hidden_size,
    "all_words":all_words,
    "tags":tags
}

FILE="s_data.pth"
torch.save(data,FILE)
print(f'Training Complete. File saved to {FILE}')