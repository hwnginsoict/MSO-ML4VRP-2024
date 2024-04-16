import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 
from Local_search import *

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

def ls_1route(ants_route, colony):
    index = np.random.randint(0, len(ants_route.values()))
    route = ants_route[index]
    colony_distance = caculate_distance(route, colony)
    distance = 0
    for value in ants_route.values():
        distance += caculate_distance(value, colony)
    for i in range(1, len(route) - 1):
        test_route = copy.deepcopy(route)
        test_route.remove(route[i])

        for j in range(1, len(test_route) - 1):
            test_route_2 = copy.deepcopy(test_route)
            test_route_2.insert(j, route[i])
            if check(route_1_ver2(test_route_2)):
                if caculate_distance(test_route_2, colony) < colony_distance:
                    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
                    ants_route_copy[index] = test_route_2
                    distance = 0
                    for value in ants_route_copy.values():
                        distance += caculate_distance(value, colony)
                    return distance, change(ants_route_copy)
    return distance, change(ants_route)

    
def destroy_ls_1route(ants_route, colony):
    ants_route_copy = {key: copy.deepcopy(value) for key, value in ants_route.items()}
    indexs = np.random.choice(np.arange(len(ants_route.values())), size=3, replace=True)
    for index in indexs:
        route = ants_route[index]
        for i in range(1, len(route) - 1):
            test_route = copy.deepcopy(route)
            test_route.remove(route[i])
            for j in range(1, len(test_route) - 1):
                test_route_2 = copy.deepcopy(test_route)
                test_route_2.insert(j, route[i])
                if check(route_1_ver2(test_route_2)):
                        ants_route_copy[index] = test_route_2
    colony_distance = 0
    for value in ants_route_copy.values():
        colony_distance += caculate_distance(value, colony)
    return colony_distance, change(ants_route_copy)





def destroy(ants_route, colony, p):
    index = np.random.randint(0, len(ants_route.values()))
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
    return travel_distance, change(ants_route_copy)
    

