from Load_data import *
from Model.Config import *
import torch 
from Model.Gen_CVRPTW_data import *

device = None 

def normalize_data(cfg):
    max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
    data = torch.tensor([[float(x) for x in y] for y in data])
    tsp_coordinates = data[:, 1:3] / torch.max(data[:, 1:3]) * cfg.time_factor # ok
    demands = torch.tensor(demand, dtype = torch.float32)/max_cap * cfg.capacity # ok
    time_window = data[:, 4:] / torch.max(data[:, 1:3]) * cfg.time_factor # ok
    durations = time_window[:, -1] 
    service_window = cfg.service_window
    time_factor = cfg.time_factor
    distances = gen_distance_matrix(tsp_coordinates, device = device)


    pyg_data_normalize = gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device, scale = 1000.0)
    return pyg_data_normalize

