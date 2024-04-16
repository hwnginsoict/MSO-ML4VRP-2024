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
import math 
import numpy as np
from Model.Config import *
from Model.AMO import * 

# cfg = Data_100()
# device = None


class ACO():

    def __init__(self,  # 0: depot
                 distances, # (n, n)
                 demand,   # (n, )
                 time_window, # (n, 3)
                 k,
                 topk,
                 capacity,
                 n_ants=10,
                 decay=0.9,
                 alpha=2,
                 beta=0.5,
                 elitist=False,
                 min_max=False,
                 pheromone=None,
                 heuristic=None,
                 min=None,
                 device='cpu',
                 adaptive=False,
                 prob = 0.3
                 ): # DONE
        self.prob = prob 
        self.topk = topk,
        self.k = k
        self.time_window = time_window
        self.problem_size = len(distances)
        self.distances = distances
        self.capacity = capacity
        self.demand = demand
        self.n_ants = n_ants
        self.decay = decay
        self.alpha = alpha
        self.beta = beta
        self.elitist = elitist or adaptive
        self.min_max = min_max
        self.adaptive = adaptive


        if pheromone is None:
            self.pheromone = torch.ones_like(self.distances, device = device)
        else:
            self.pheromone = pheromone

        self.heuristic = torch.where(distances == 0, -1e10, 1/distances) if heuristic is None else heuristic # TODO
        # if self.heuristic is None:
        #     for i in len(d)

        self.shortest_path = None
        self.lowest_cost = float('inf')

        self.device = device

    def sample(self): # DONE
        paths = self.gen_path(require_start_moves=True)
        costs = self.gen_path_costs(paths)
        return paths, costs


    def run(self): # DONE
        paths_1 = self.gen_path(require_start_moves=True)
        costs_1 = self.gen_path_costs(paths_1)
        paths_2 = self.gen_path(require_start_moves=False)
        costs_2 = self.gen_path_costs(paths_2)
        c = max(paths_1.shape[0], paths_2.shape[0]) + 1
        paths_1 = torch.concat((paths_1, torch.zeros(size = (c - paths_1.shape[0], paths_1.shape[1]))), dim = 0)
        paths_2 = torch.concat((paths_2, torch.zeros(size = (c - paths_2.shape[0], paths_2.shape[1]))), dim = 0)
        paths = torch.concat((paths_1, paths_2), dim = 1)
        costs = torch.concat((costs_1, costs_2), dim = 0)
        # best_cost, best_idx = costs.min(dim=0)
        # if best_cost < self.lowest_cost:
        #     self.shortest_path = paths[:, best_idx]
        #     self.lowest_cost = best_cost
        #     improved = True
        # self.update_pheronome(paths, costs)
        return paths, costs

        # return self.lowest_cost

    @torch.no_grad()
    def update_pheronome(self, paths, costs): # DONE (Lấy 10% tốt nhất)
        '''
        Args:
            paths: torch tensor with shape (problem_size, n_ants)
            costs: torch tensor with shape (n_ants,)
        '''
        self.pheromone = self.pheromone * self.decay
        paths = paths.to(torch.long)
        for i in range(self.n_ants):
            path = paths[:, i]
            cost = costs[i]
            # print(path[:-1])
            self.pheromone[path[:-1], torch.roll(path, shifts=-1)[:-1]] += 1.0/cost
        
        best_cost, best_idx = costs.min(dim=0)
        best_tour = paths[:, best_idx]
        self.pheromone[best_tour[:-1], torch.roll(best_tour, shifts=-1)[:-1]] += 1.0/best_cost


        self.pheromone[self.pheromone < 1e-10] = 1e-10

    @torch.no_grad()
    def gen_path_costs(self, paths): # DONE
        u = paths.permute(1, 0) # shape: (n_ants, max_seq_len)
        v = torch.roll(u, shifts=-1, dims=1)
        return torch.sum(self.distances[u[:, :-1], v[:, :-1]], dim=1)

    def gen_path(self, require_start_moves): # DONE
        actions = torch.zeros((self.n_ants * self.k,), dtype=torch.long, device=self.device)

        visit_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        visit_mask = self.update_visit_mask(visit_mask, actions)
        used_capacity = torch.zeros(size=(self.n_ants * self.k,), device=self.device)
        used_time = torch.zeros(size=(self.n_ants * self.k,), device=self.device)

        used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
        used_time, time_mask = self.update_time_mask(actions, actions, used_time)


        paths_list = [actions] # paths_list[i] is the ith move (tensor) for all ants


        done = self.check_done(visit_mask, actions)
        # first_start
        if require_start_moves == True:
            for _ in range(1):
                pre_node = copy.deepcopy(actions)
                actions = self.topk_start_move()
                paths_list.append(actions)
                visit_mask = self.update_visit_mask(visit_mask, actions)
                used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
                used_time, time_mask = self.update_time_mask(actions, pre_node, used_time)

                done = self.check_done(visit_mask, actions)

        while not done:
            pre_node = copy.deepcopy(actions)
            actions = self.pick_move(actions, visit_mask, capacity_mask, time_mask)
            paths_list.append(actions)
            visit_mask = self.update_visit_mask(visit_mask, actions)
            used_capacity, capacity_mask = self.update_capacity_mask(actions, used_capacity)
            used_time, time_mask = self.update_time_mask(actions,pre_node, used_time)


            done = self.check_done(visit_mask, actions)
        else:
            return torch.stack(paths_list)


    def topk_start_move(self): # DONE
        actions = self.topk[0].repeat(self.n_ants) # (n_ants * k, )
        return actions

    def pick_move(self, prev, visit_mask, capacity_mask, time_mask): # DONE
        
        pheromone = self.pheromone[prev].to(device) # shape: (n_ants, p_size)
        heuristic = self.heuristic[prev].to(device) # shape: (n_ants, p_size)
        dist = ((pheromone ** self.alpha) * (heuristic ** self.beta) * visit_mask *  capacity_mask * time_mask) # shape: (n_ants, p_size)
        r = torch.rand(1)
        if r < self.prob:
            dist_copy = torch.where(dist == 0, -1e20, dist).to(device)
            dist_1 = Categorical(logits = dist_copy)

            actions = dist_1.sample() # shape: (n_ants,)
        else:
            dist_copy = torch.where(dist == 0, -1e20, dist).to(device)
            dist_copy += 100
            actions = torch.argmax(dist_copy, dim = 1)

        return actions

    def update_visit_mask(self, visit_mask, actions): # DONE
        visit_mask[torch.arange(self.n_ants * self.k, device=self.device), actions] = 0
        visit_mask[:, 0] = 1 # depot can be revisited with one exception
        visit_mask[(actions==0) * (visit_mask[:, 1:]!=0).any(dim=1), 0] = 0 # one exception is here
        return visit_mask

    def update_time_mask(self, cur_nodes, pre_nodes, used_time): # DONE
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_time: shape (n_ants, )
            time_mask: shape (n_ants, p_size)
        Returns:
            ant_time: updated capacity
            time_mask: updated mask
        '''
        time_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        # update time
        used_time = used_time + self.distances[pre_nodes,cur_nodes]
        used_time[cur_nodes==0] = 0
        start = self.time_window[cur_nodes, 0]
        used_time = torch.where(used_time < start, start, used_time)
        used_time = used_time + self.time_window[cur_nodes, [2] * self.n_ants * self.k]

        # update time mask
        time = self.distances[cur_nodes.expand([self.problem_size,-1]).T.flatten(),  torch.arange(self.problem_size).repeat(1,self.n_ants * self.k)].view(self.n_ants * self.k, self.problem_size).to(device)
        # (self.n_ants * self.k, self.problem_size)
        time = used_time.view(-1,1).expand(-1, self.problem_size) + time
        finish = self.time_window[:, 1].expand(self.n_ants * self.k, -1)
        time_mask[time > finish] = 0
        return used_time, time_mask


    def update_capacity_mask(self, cur_nodes, used_capacity): # DONE
        '''
        Args:
            cur_nodes: shape (n_ants, )
            used_capacity: shape (n_ants, )
            capacity_mask: shape (n_ants, p_size)
        Returns:
            ant_capacity: updated capacity
            capacity_mask: updated mask
        '''
        capacity_mask = torch.ones(size=(self.n_ants * self.k, self.problem_size), device=self.device)
        # update capacity
        used_capacity[cur_nodes==0] = 0
        used_capacity = used_capacity + self.demand[cur_nodes]
        # update capacity_mask
        remaining_capacity = self.capacity - used_capacity # (n_ants,)
        self.remain = remaining_capacity
        remaining_capacity_repeat = remaining_capacity.unsqueeze(-1).repeat(1, self.problem_size).to(device) # (n_ants, p_size)
        demand_repeat = self.demand.unsqueeze(0).repeat(self.n_ants * self.k, 1).to(device) # (n_ants, p_size)
        self.used_cap = used_capacity
        capacity_mask[demand_repeat > remaining_capacity_repeat] = 0
        return used_capacity, capacity_mask

    def check_done(self, visit_mask, actions): # DONE
        return (visit_mask[:, 1:] == 0).all() and (actions == 0).all()
    

from Load_data import *
from Model.Config import *
import torch 
from Model.Gen_CVRPTW_data import *

# device = None 
# model = Net3()
# model.load_state_dict(torch.load('AMO_ACO_{}.pt'.format(cfg.graph_size), map_location=torch.device('cpu')))
# model.to(device)

# max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
# data = torch.tensor([[float(x) for x in y] for y in data])
# tsp_coordinates = data[:, 1:3] / torch.max(data[:, 1:3]) * cfg.time_factor # ok
# demands = torch.tensor(demand, dtype = torch.float32)/max_cap * cfg.capacity # ok
# time_window = data[:, 4:] / torch.max(data[:, 1:3]) * cfg.time_factor # ok
# durations = time_window[:, -1] 
# service_window = cfg.service_window
# time_factor = cfg.time_factor
# distances = gen_distance_matrix(tsp_coordinates, device = device)
# pyg_data = gen_pyg_data(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
# pyg_data_normalize = gen_pyg_data_normalize(cfg, demands, time_window, durations, service_window, time_factor, distances, device = device)
# heuristic_measure, log, topk = model(pyg_data_normalize)
# heuristic_measure = heuristic_measure.reshape((cfg.graph_size+1, cfg.graph_size+1))


# max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
# data = torch.tensor([[float(x) for x in y] for y in data])
# tsp_coordinates = data[:, 1:3] 
# demands = torch.tensor(demand, dtype = torch.float32)
# time_window = data[:, 4:]
# durations = time_window[:, -1] 
# distances = gen_distance_matrix(tsp_coordinates, device = device)




# aco = ACO(distances, demands, time_window, 10, topk, max_cap, heuristic=heuristic_measure)
# paths, cost = aco.run()
# print(torch.mean(cost))
# paths, cost = aco.sample()
# print(paths)
# print('--------')







    

