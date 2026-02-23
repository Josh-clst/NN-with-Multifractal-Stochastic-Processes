import sys
sys.path.insert(0, '.')

import functions
import numpy as np
import torch
import matplotlib.pyplot as plt
import torch.nn.functional as F

# Données cibles
data = np.load("Train_TurbModane_N65536_sampling4_Nreal256_v3.npz")
S2 = torch.from_numpy(data['S2'].mean(axis=0)).float()
flatness = torch.from_numpy(data['Flat'].mean(axis=0)).float()
scales = torch.from_numpy(data["scales"]).float()

torch.manual_seed(0)