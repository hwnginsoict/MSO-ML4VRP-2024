import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 

def find_route_target(customer, ants_route, colony, p, index):
    lim = p * max(colony.distance_matrix.values())
    route_target = []
    for key, value in ants_route.items():
        if key != index:
            check = 1
            for customer_target in value[1:-1]:
                if colony.distance_matrix[customer, customer_target] > lim:
                    check = 0
                    break
            if check:
                route_target.append(key)
    return route_target



def injection(ants_route, colony, p):
    min_route = 1000
    index = 0
    for key, value in ants_route.items():
        if len(value) < min_route and len(value) > 2:
            index = key
            min_route = len(value)
    colony_distance = 0
    for value in ants_route.values():
        colony_distance += caculate_distance(value, colony)
    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
    select = []
    for customer in ants_route[index][1:-1]:
        route_target = find_route_target(customer, ants_route_copy, colony, p, index)
        for route in route_target:
            done = 0
            new_route = check_feasible(customer, ants_route_copy[route], colony)
            if new_route != []:
                ants_route_copy[route] = new_route
                select.append(customer)
                done = 1
                break


            else:
                for customer_target in ants_route_copy[route][1:-1]:
                    new_route = ants_route_copy[route].copy()
                    new_route.remove(customer_target)
                    r = check_feasible(customer, new_route, colony)
                    if r == []:
                        continue

                    check = 0

                    for key, value in ants_route_copy.items():
                        if key != index and key != route:
                            test = check_feasible(customer_target, value, colony)
                            if test != []:
                                ants_route_copy[key] = test 
                                check = 1
                                break 
                    if check:
                        ants_route_copy[route] = r 
                        # select.append(customer)
                        done = 1
                        break 
            if done:
                select.append(customer)
                break 



    for i in select:
        ants_route_copy[index].remove(i)
    
    travel_distance = 0
    for route in ants_route_copy.values():
        for i in range(len(route)-1):
            travel_distance += colony.distance_matrix[route[i], route[i+1]]
    if travel_distance < colony_distance:
        return travel_distance, change(ants_route_copy)
    
    return colony_distance, change(ants_route)