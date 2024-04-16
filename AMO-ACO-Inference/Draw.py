import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 


def draw(ants_route, colony, require_route = True):
    data = colony.data
    coord = np.array(data, dtype=np.float64)[:, 1:3]
    n_routes = len(ants_route.keys())
    cmap = plt.cm.get_cmap('hsv')
    colors = [cmap(i / n_routes) for i in range(n_routes)]
    
    for key, route in ants_route.items():
        color = colors[key]
        if require_route:
            for j in range(len(route) - 1):
                x1, y1 = coord[route[j]-1]
                x2, y2 = coord[route[j+1]-1]
                # plt.scatter([x1, x2], [y1, y2], color=color, marker='s')
                if j == len(route) - 2:
                    plt.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1, label = f'Route {key}')
                else:
                    plt.plot([x1, x2], [y1, y2], color=color, linestyle='-', linewidth=1)
        
        for point in route[1:-1]:
            x, y = coord[point-1]
            plt.scatter(x, y, color=color, marker='^')
        
        

    depot_x, depot_y = coord[0]
    plt.scatter(depot_x, depot_y, color='black', marker='o', s=100)
    plt.grid()
    plt.title('Time_Window_CVRP')
    plt.legend()
    plt.show()




    
    