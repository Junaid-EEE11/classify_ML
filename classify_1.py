from sklearn.datasets import make_blobs
NUM_CLASSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 42
X, y = make_blobs(n_samples=1000, n_features=NUM_FEATURES, centers=NUM_CLASSES, cluster_std=1.5, random_state=RANDOM_SEED)


import torch
from torch import nn
device = "cuda" if torch.cuda.is_available() else "cpu"

X = torch.from_numpy(X).type(torch.float)
y = torch.from_numpy(y).type(torch.float)

def relu(x):
  return torch.maximum(torch.tensor(0), x)

def sigmoid(x):
  return 1 / (1 + torch.exp(-x))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42)

class cls01(nn.Module):
    def __init__(self, input_features, output_features, hidden_units=10):
        super().__init__()
        self.layer_1 = nn.Linear(in_features=2, out_features=10)
        self.layer_2 = nn.Linear(in_features=10, out_features=4)
        #self.relu = relu() #nn.ReLU()
        #self.sigmoid = nn.Sigmoid()
        #self.linear_layer_stack = nn.Sequential(
            #nn.Linear(in_features=input_features, out_features=hidden_units),
            # nn.ReLU(), 
            #nn.Linear(in_features=hidden_units, out_features=hidden_units),
            # nn.ReLU(),
            #nn.Linear(in_features=hidden_units, out_features=output_features),
      
    def forward(self, x):
        #return self.layer_2(self.relu(self.layer_1(x)))
        #return self.linear_layer_stack(x)
        return self.layer_2(relu(self.layer_1(x)))

model_0 = cls01(input_features=2,output_features=4, hidden_units=10).to(device)
loss_fn = nn.CrossEntropyLoss() #nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(), lr=0.1)

def accuracy_fn(y_true, y_pred):
    correct = torch.eq(y_true, y_pred).sum().item()
    acc = (correct / len(y_pred)) * 100 
    return acc

torch.manual_seed(42)
epochs = 1000

X_train, y_train = X_train.to(device), y_train.to(device)
X_test, y_test = X_test.to(device), y_test.to(device)

for epoch in range(epochs):
    model_0.train()
    y_logits = model_0(X_train).squeeze()
    y_pred = torch.softmax(y_logits, dim=1).argmax(dim=1)  #y_pred = torch.round(torch.sigmoid(y_logits))
    loss = loss_fn(y_logits, y_train) 
    acc = accuracy_fn(y_true=y_train, y_pred=y_pred) 
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(X_blob_test)  #test_logits = model_0(X_test).squeeze() 
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)  #test_pred = torch.round(torch.sigmoid(test_logits))
        test_loss = loss_fn(test_logits, y_test)
        test_acc = accuracy_fn(y_true=y_test, y_pred=test_pred)
    if epoch % 10 == 0:
        print(f"Epoch: {epoch} | Loss: {loss:.5f}, Accuracy: {acc:.2f}% | Test loss: {test_loss:.5f}, Test acc: {test_acc:.2f}%")



