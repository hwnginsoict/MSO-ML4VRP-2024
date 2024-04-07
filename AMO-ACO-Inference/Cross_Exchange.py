import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 

def check_constraint(route, colony):
    service_time = 0
    CAPACITY = colony.capacity 
    new_route = route.copy()
    m = len(route) - 1
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
        return True
    return False 




def cross_exchange(ants_route, colony):
    lst = []
    keys = []
    # print(colony.travel_distance)
    # print(ants_route)
    ans = 0
    for key, value in ants_route.items():
        distance = 0
        for i in range(len(value)-1):
            distance += colony.distance_matrix[value[i], value[i+1]]
        ans += distance
        lst.append(distance)
        keys.append(key)
    # print(distance)
    select_1, select_2 = np.argsort(np.array(lst))[-2:]
    segment_1 = np.random.randint(0,3)
    segment_2 = np.random.randint(0,3)
    if np.random.random() < 0.5:
        route_1 = ants_route[select_1]
        route_2 = ants_route[select_2]
    else:
        select_1, select_2 = central_2([value for value in ants_route.values()], colony)
        route_1 = ants_route[select_1]
        route_2 = ants_route[select_2]
    m = len(route_1)
    n = len(route_2)
    result = []
    distance = caculate_distance(route_1, colony) + caculate_distance(route_2, colony)
    distance_2 = []
    for i in range(1, m-segment_1):
        for j in range(1, n-segment_2):
            lst_1 = copy.deepcopy(route_1)
            lst_2 = copy.deepcopy(route_2)
            lst_1[i:i+segment_1] = route_2[j:j+segment_2]
            lst_2[j:j+segment_2] = route_1[i:i+segment_1]
            if check_constraint(lst_1, colony) and check_constraint(lst_2, colony):
                result.append((lst_1, lst_2))
                distance_2.append(caculate_distance(lst_1, colony) + caculate_distance(lst_2, colony))
    if distance_2 != []:
        if min(distance_2) < distance:
            ants_route_copy = ants_route.copy()
            lst_1, lst_2 = result[np.argsort(np.array(distance_2))[0]]
            ants_route_copy[select_1] = lst_1 
            ants_route_copy[select_2] = lst_2
            travel_distance = 0
            for value in ants_route_copy.values():
                travel_distance += caculate_distance(value, colony)
            return travel_distance, change(ants_route_copy)
    # return colony.travel_distance, change(ants_route)
    return ans, change(ants_route)
