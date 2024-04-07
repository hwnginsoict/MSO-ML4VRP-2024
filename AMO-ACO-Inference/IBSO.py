import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 

def individual(data, colony, CAP, heuristic_measure):
    colony_1 = Ant(data,CAP,0.8, heuristic_measure)
    colony_1.customer_cord()
    colony_1.euclidean_distance()
    colony_1.width_window()
    colony_1.pheromon = colony.pheromon

    colony_1.travel_distance = 0
    ants_travels={}
    ants_route={}
    travels=[]
    path=[1]
    i=0
    colony_1.visited_list = [1]
    while True:
        colony_1.make_candidate_list()
        colony_1.choose_next_node()
        colony_1.move()
        path.append(colony_1.next_node)
        travel=colony_1.travel
        travels.append(travel)
        if travel[1]==1:
            if travel==(1,1):
                break
            else:
                ants_travels[i]=travels
                ants_route[i]=path
                        
                path=[1]
                travels=[]
                i=i+1
                colony_1.current_point=1
                colony_1.capacity=CAP
                colony_1.service_time=0
    return colony_1.travel_distance, ants_route



def IBSO(population, distance, colony, CAP, heuristic_measure, p = 0.01):
    bound = (min(distance) + max(distance))/2
    team_A = []
    cost_team_A = []
    team_B = []
    cost_team_B = []
    for i, ants_route in enumerate(population):
        if distance[i] > bound:
            team_A.append(ants_route)
            cost_team_A.append(distance[i])
        else:
            team_B.append(ants_route)
            cost_team_B.append(distance[i])
    team_A_new = []
    cost_team_A_new = []
    for i in team_A:
        travel_distance, route = individual(colony.data, colony, CAP, heuristic_measure)
        team_A_new.append(route)
        cost_team_A_new.append(travel_distance)
    r = np.random.random()
    travel_distance, route = individual(colony.data, colony, CAP, heuristic_measure)
    if r < p:
        if cost_team_B[np.argsort(np.array(cost_team_B))[0]] > travel_distance:
            team_B[np.argsort(np.array(cost_team_B))[0]] = route
            cost_team_B[np.argsort(np.array(cost_team_B))[0]] = travel_distance
    else:
        id = np.random.randint(0,len(team_B))
        if cost_team_B[id] > travel_distance:
            team_B[id] = route
            cost_team_B[id] = travel_distance
    return team_A_new + team_B, cost_team_A_new + cost_team_B