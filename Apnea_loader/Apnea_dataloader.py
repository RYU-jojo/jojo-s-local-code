# import pandas as pd
# import numpy as np
# from torch.utils.data import Dataset, DataLoader
# import torch

# class ApneaDataset(Dataset):
#     def __init__(self, data_file, labels_file, seq_len):
#         self.data = pd.read_csv(data_file)
#         self.labels = pd.read_csv(labels_file)
#         self.seq_len = seq_len
#         self.num_sequences = len(self.data) // seq_len
#         data_file = '/home/ubuntu22/Time-Series-Library-main/datasets/Apnea/Apnea_23/01/PSG/psgresp20.csv'
#         labels_file = '/home/ubuntu22/Time-Series-Library-main/datasets/Apnea/Apnea_23/01/PSG/psgevents.csv'
#         seq_len = 600
#         Dataset = ApneaDataset(data_file, labels_file, seq_len, shuffle=True)
#         assert self.num_sequences == len(self.labels),"Number of sequences does not match number of labels"
    
#     def __len__(self):
#         return self.num_sequences
    
#     def __getitem__(self, idx):
#         start_idx = idx * self.seq_len
#         end_idx = start_idx + self.seq_len
#         sequence = self.data.iloc[start_idx:end_idx].values
#         label = self.labels.iloc[idx].value[0]
#         return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)

import os
import pandas as pd
import torch 
from torch.utils.data import Dataset, DataLoader

class Dpnea_dataset(Dataset):
    def __init__(self, root_dir, seq_len):
        self.root_dir = root_dir
        self.seq_len = seq_len
        self.data_files = []
        self.label_files = []
        
        for folder_name in sorted(os.listdir(root_dir)):
            folder_path = os.path.join(root_dir, folder_name)
            if os.path.isdir(folder_path):
                data_file = os.path.join(folder_path, 'psgresp20.csv')
                label_file = os.path.join(folder_path, 'psgevents.csv')
                if os.path.exists(data_file) and os.path.exists(label_file):
                    self.data_files.append(data_file)
                    self.label_files.append(label_file)
        
        self.num_seq = 0
        self.total_labels = 0
        
        for data_file, label_file in (self.data_files, self.label_files):
            data_len = len(pd.read_csv(data_file)) // seq_len
            label_len = len(pd.read_csv(label_file))
            self.num_seq += data_len
            self.total_labels += label_len
            
            assert data_len == label_len, f"Number of sequences ({data_length}) does not match number of labels ({label_len}) in file {data_file}"
        assert self.num_seq == self.total_labels, "Total number of sequences does not match total number of labels"
        
    def __len__(self):
        return self.num_seq
    
    def __getitem__(self, idx):
        cumulative_seq = 0
        for i, data_file in enumerate(self.data_files):
            data_len = len(pd.read_csv(data_file)) // self.seq_len
            if cumulative_seq + data_len > idx:
                file_idx = i
                seq_idx = idx - cumulative_seq
                break
            cumulative_seq += data_len
            
        data = pd.read_csv(self.data_files[file_idx])
        labels = pd.read_csv(self.label_files[file_idx])
        
        start_idx = seq_idx * self.seq_len
        end_idx = start_idx + self.seq_len
        sequence = data.iloc[start_idx:end_idx].values
        label = labels.iloc[seq_idx].value[0]
        
        return torch.tensor(sequence, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
            