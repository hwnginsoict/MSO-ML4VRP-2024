import torch
from torch.distributions import Categorical
import random
import itertools
import numpy as np
import copy
import torch
from torch import nn
from torch.nn import functional as F
from copy import deepcopy
from .Gen_CVRPTW_data import *
from .Model_Heuristic_Measure import *
from .GraphAttetionEncoder import *
import math 
import numpy as np
from .Model_predict_start_node import *
import time
from .Config import *
import matplotlib.pyplot as plt 
from .AMO import *

EPS = 1e-10
lr = 3e-4
device = None

model  = Net3().to(device)
cfg = Data_200()


model.train()
CVRPTW = generate_cvrptw_data(cfg)[0]
tsp_coordinates = torch.cat((CVRPTW.depot_loc.expand(1,-1), CVRPTW.node_loc), dim = 0)
demands = torch.cat((torch.tensor([0]).to(device), CVRPTW.demand), dim = 0)
time_window = torch.cat((CVRPTW.depot_tw.expand(1,-1), CVRPTW.node_tw), dim = 0)
durations = torch.cat((torch.tensor([0]).to(device), CVRPTW.durations), dim = 0)
time_window = torch.cat((time_window, durations.view(-1,1)), dim = 1)
service_window = CVRPTW.service_window
time_factor = CVRPTW.time_factor
distances = gen_distance_matrix(tsp_coordinates, device = device)
pyg_data = gen_pyg_data(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
pyg_data_normalize = gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
heuristic_measure, log, topk = model(pyg_data_normalize)
heuristic_measure = heuristic_measure.reshape((cfg.graph_size+1, cfg.graph_size+1)) + EPS
print(heuristic_measure.shape)