import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as D
import tables as tb
from sklearn.metrics import (matthews_corrcoef, 
                             confusion_matrix, 
                             f1_score, 
                             roc_auc_score,
                             accuracy_score,
                             roc_auc_score)


# set the device to GPU if available
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
MAIN_PATH = '.'
DATA_FILE = 'mt_data.h5'
MODEL_FILE = 'chembl_mt.model'
N_WORKERS = 8 # Dataloader workers, prefetch data in parallel to have it ready for the model after each batch train
BATCH_SIZE = 32 # https://twitter.com/ylecun/status/989610208497360896?lang=es
LR = 2 # Learning rate. Big value because of the way we are weighting the targets
N_EPOCHS = 2 # You should train longer!!!

class ChEMBLDataset(D.Dataset):
    
    def __init__(self, file_path):
        self.file_path = file_path
        with tb.open_file(self.file_path, mode='r') as t_file:
            self.length = t_file.root.fps.shape[0]
            self.n_targets = t_file.root.labels.shape[1]
        
    def __len__(self):
        return self.length
    
    def __getitem__(self, index):
        with tb.open_file(self.file_path, mode='r') as t_file:
            structure = t_file.root.fps[index]
            labels = t_file.root.labels[index]
        return structure, labels


dataset = ChEMBLDataset(f"{MAIN_PATH}/{DATA_FILE}")
validation_split = .2
random_seed= 42

dataset_size = len(dataset)
indices = list(range(dataset_size))
split = int(np.floor(validation_split * dataset_size))

np.random.seed(random_seed)
np.random.shuffle(indices)
train_indices, test_indices = indices[split:], indices[:split]

train_sampler = D.sampler.SubsetRandomSampler(train_indices)
test_sampler = D.sampler.SubsetRandomSampler(test_indices)

# dataloaders can prefetch the next batch if using n workers while
# the model is tranining
train_loader = torch.utils.data.DataLoader(dataset,
                                           batch_size=BATCH_SIZE,
                                           num_workers=N_WORKERS,
                                           sampler=train_sampler)

test_loader = torch.utils.data.DataLoader(dataset, 
                                          batch_size=BATCH_SIZE,
                                          num_workers=N_WORKERS,
                                          sampler=test_sampler)


class ChEMBLMultiTask(nn.Module):
    """
    Architecture borrowed from: https://arxiv.org/abs/1502.02072
    """
    def __init__(self, n_tasks):
        super(ChEMBLMultiTask, self).__init__()
        self.n_tasks = n_tasks
        self.fc1 = nn.Linear(1024, 2000)
        self.fc2 = nn.Linear(2000, 100)
        self.dropout = nn.Dropout(0.25)

        # add an independet output for each task int the output laer
        for n_m in range(self.n_tasks):
            self.add_module(f"y{n_m}o", nn.Linear(100, 1))
        
    def forward(self, x):
        h1 = self.dropout(F.relu(self.fc1(x)))
        h2 = F.relu(self.fc2(h1))
        out = [torch.sigmoid(getattr(self, f"y{n_m}o")(h2)) for n_m in range(self.n_tasks)]
        return out
    
# create the model, to GPU if available
model = ChEMBLMultiTask(dataset.n_targets).to(device)

# binary cross entropy
# each task loss is weighted inversely proportional to its number of datapoints, borrowed from:
# http://www.bioinf.at/publications/2014/NIPS2014a.pdf
with tb.open_file(f"{MAIN_PATH}/{DATA_FILE}", mode='r') as t_file:
    weights = torch.tensor(t_file.root.weights[:])
    weights = weights.to(device)

criterion = [nn.BCELoss(weight=w) for x, w in zip(range(dataset.n_targets), weights.float())]

# stochastic gradient descend as an optimiser
optimizer = torch.optim.SGD(model.parameters(), LR)

# model is by default in train mode. Training can be resumed after .eval() but needs to be set to .train() again
model.train()
for ep in range(N_EPOCHS):
    for i, (fps, labels) in enumerate(train_loader):
        # move it to GPU if available
        fps, labels = fps.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(fps)
        
        # calc the loss
        loss = torch.tensor(0.0).to(device)
        for j, crit in enumerate(criterion):
            # mask keeping labeled molecules for each task
            mask = labels[:, j] >= 0.0
            if len(labels[:, j][mask]) != 0:
                # the loss is the sum of each task/target loss.
                # there are labeled samples for this task, so we add it's loss
                loss += crit(outputs[j][mask], labels[:, j][mask].view(-1, 1))

        loss.backward()
        optimizer.step()

        if (i+1) % 500 == 0:
            print(f"Epoch: [{ep+1}/{N_EPOCHS}], Step: [{i+1}/{len(train_indices)//BATCH_SIZE}], Loss: {loss.item()}")
    
y_trues = []
y_preds = []
y_preds_proba = []

# do not track history
with torch.no_grad():
    for fps, labels in test_loader:
        # move it to GPU if available
        fps, labels = fps.to(device), labels.to(device)
        # set model to eval, so will not use the dropout layer
        model.eval()
        outputs = model(fps)
        for j, out in enumerate(outputs):
            mask = labels[:, j] >= 0.0
            mask = mask.to(device)
            y_pred = torch.where(out[mask].to(device) > 0.5, torch.ones(1).to(device), torch.zeros(1).to(device)).view(1, -1)

            if y_pred.shape[1] > 0:
                for l in labels[:, j][mask].long().tolist():
                    y_trues.append(l)
                for p in y_pred.view(-1, 1).tolist():
                    y_preds.append(int(p[0]))
                for p in out[mask].view(-1, 1).tolist():
                    y_preds_proba.append(float(p[0]))

tn, fp, fn, tp = confusion_matrix(y_trues, y_preds).ravel()
sens = tp / (tp + fn)
spec = tn / (tn + fp)
prec = tp / (tp + fp)
f1 = f1_score(y_trues, y_preds)
acc = accuracy_score(y_trues, y_preds)
mcc = matthews_corrcoef(y_trues, y_preds)
auc = roc_auc_score(y_trues, y_preds_proba)

print(f"accuracy: {acc}, auc: {auc}, sens: {sens}, spec: {spec}, prec: {prec}, mcc: {mcc}, f1: {f1}")
print(f"Not bad for only {N_EPOCHS} epochs!")

torch.save(model.state_dict(), f"./{MODEL_FILE}")
"""
In [15]: dummy = torch.tensor([[0.5 for _ in range(1024)]],
    ...: dtype=torch.float32).to(device)

In [16]: onnx.export(model, dummy, path, input_names=['input1'],output_n
    ...: ames=['output1'])
"""
