import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from Ant import Ant

def check_feasible(customer, route, colony):
    m = len(route)
    for i in range(1,m):
            service_time = 0
            CAPACITY = colony.capacity 
            new_route = route.copy()
            new_route.insert(i, customer)
            cnt = 0
            for j in range(m):
                
                if (float(colony.data[new_route[j+1] - 1][3])) <= CAPACITY and (service_time + float(colony.distance_matrix[new_route[j], new_route[j+1]]) <= float(colony.data[new_route[j+1] - 1][5])):
                    cnt += 1
                    CAPACITY -= float(colony.data[new_route[j+1] - 1][3])
                    if service_time + float(colony.distance_matrix[new_route[j], new_route[j+1]]) < float(colony.data[new_route[j+1] - 1][4]):
                        service_time = float(colony.data[new_route[j+1] - 1][4]) + float(colony.data[new_route[j+1]-1][6])
                    else:
                        service_time += float(colony.distance_matrix[new_route[j], new_route[j+1]]) + float(colony.data[new_route[j+1]-1][6])
                else:
                    break
            if cnt == m:
                return new_route 
    return []



def split_route(ants_route):
     result = []
     for route in ants_route.values():
          for i in range(len(route)-1):
               result.append((route[i], route[i+1]))
     return result


def change(ants_route):
    index = 0
    lst = {}
    for value in ants_route.values():
        if value != [1,1]:
            lst[index] = value
            index += 1
    return lst 


def caculate_distance(route, colony):
    distance = 0
    for i in range(len(route)-1):
        distance += colony.distance_matrix[route[i], route[i+1]]
    return distance


def central(route, colony):
    coord = []
    for value in route:
        coord.append([np.mean([np.array(float(colony.data[i][1])) for i in value]), np.mean([np.array(float(colony.data[i][2])) for i in value])])
    distance = np.array(cdist(np.array(coord), np.array(coord), 'euclidean'))
    for i in range(len(distance)):
        distance[i][i] = 99999999.99999
    selected = np.unravel_index(np.argmin(distance), distance.shape)
    return selected 


def central_2(route, colony):
    coord = []
    for value in route:
        coord.append([np.mean([np.array(float(colony.data[i-1][1])) for i in value]), np.mean([np.array(float(colony.data[i-1][2])) for i in value])])
    distance = np.array(cdist(np.array(coord), np.array(coord), 'euclidean'))
    for i in range(len(distance)):
        distance[i][i] = 99999999.99999
    selected = np.unravel_index(np.argmin(distance), distance.shape)
    return selected 


