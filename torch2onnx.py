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
from torch import onnx

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

dataset = ChEMBLDataset(f"{MAIN_PATH}/{DATA_FILE}")
validation_split = .2
random_seed= 42

dataset_size = len(dataset)
model = ChEMBLMultiTask(dataset.n_targets).to(device)
path = './model_onnx.onnx'
dummy = torch.tensor([[0.5 for _ in range(1024)]],
dtype=torch.float32).to(device)
onnx.export(model, dummy, path, input_names=['input_1'],output_names=['output'])
