import pandas as pd
import numpy as np
from model import LSTMClassifier
from data_loader import get_train_loader, collate_fn
import torch.nn as nn
import torch.optim
import torch.autograd as autograd
from torch.utils.data.sampler import SubsetRandomSampler

class LSTMClassifier(nn.Module):

    def __init__(self, vocab_size, hidden_dim, output_size):

        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size

        self.lstm = nn.LSTM(vocab_size, hidden_dim, num_layers=1)

        self.hidden2out = nn.Linear(hidden_dim, output_size)
        self.softmax = nn.LogSoftmax()

        self.dropout_layer = nn.Dropout(p=0.2)


    def init_hidden(self):
        return(autograd.Variable(torch.randn(1, 1, self.hidden_dim)),
                        autograd.Variable(torch.randn(1, 1, self.hidden_dim)))


    def forward(self, batch, lengths):

        self.hidden = self.init_hidden()

        packed_input = pack_padded_sequence(batch, lengths, batch_first=True)
        outputs, (ht, ct) = self.lstm(packed_input, self.hidden)

        output = self.hidden2out(ht[-1])
        output = self.softmax(output)

        return output


#Load datasets
cities = pd.read_csv("./csv/cities.csv", sep=";")
regions = pd.read_csv("./csv/regions.csv", sep=";")
regions.columns = ["regions_id", "name", "slug", "iso_code"]
departments = pd.read_csv("./csv/departments.csv", sep=";")
departments.columns = ["departments_id", "regions_id", "code", "name", "slug", "iso_code"]

#Merge datasets
temp_df = pd.merge(departments, regions, on="regions_id", how="left")
final_df = pd.merge(cities, temp_df, on="departments_id", how="left")

#Only keep cities and region ids associated
df = final_df[["regions_id", "pattern"]]
df["regions_id"] = df["regions_id"].apply(lambda x : x-1)

#Print the region names
n_categories = len(set(df["regions_id"].tolist()))
print("Number of regions:", n_categories)
print()
regions["regions_id"] = regions["regions_id"].apply(lambda x : x-1)
all_categories = regions.set_index("regions_id").to_dict()["name"]
print("Dictionary of regions and their id:", all_categories)

#Separate inputs and labels
cities = df["pattern"].tolist()
labels = df["regions_id"].tolist()

#Define a split for train/valid
valid_size = 0.2
batch_size = 10
num_train = len(cities)
indices = list(range(num_train))
split = int(np.floor(valid_size * num_train))

train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

#Load data generators
train_data_loader = get_train_loader(cities=cities, labels=labels, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, sampler=train_sampler)
valid_data_loader = get_train_loader(cities=cities, labels=labels, batch_size=batch_size, shuffle=False, collate_fn=collate_fn, sampler=valid_sampler)

#Initialize the model to train
model = LSTMClassifier(27, 10, 14)

# Loss and Optimizer
criterion = nn.NLLLoss()
learning_rate = 0.8 
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# train
losses = []

num_epochs = 10

# Train the Model
for epoch in range(num_epochs):
    print("##### epoch {:2d}".format(epoch + 1))
    for i, batch in enumerate(train_data_loader):
        city = autograd.Variable(batch[0])
        length = batch[2].cpu().numpy()
        label = batch[1].long()
        optimizer.zero_grad()
        pred = model(city, length)
        true = autograd.Variable(label)
        loss = criterion(pred, true)
        loss.backward()
        optimizer.step()

        if (i + 1) % 1000 == 0:
            print('Epoch [%d/%d], Iter [%d/%d] Loss: %.4f'
                  % (epoch + 1, num_epochs, i + 1, len(train_data_loader), loss.data[0]))

    valid_losses = 0
    for i, batch in enumerate(valid_data_loader):
        city = autograd.Variable(batch[0])
        label = autograd.Variable(batch[1].long())
        length = batch[2].cpu().numpy()
        pred = model(city, length)
        loss = criterion(pred, label)
        valid_losses += loss.data[0]

    print('Validation MSE of the model at epoch {} is: {}'.format(epoch, np.round(valid_losses / len(valid_data_loader), 2)))


