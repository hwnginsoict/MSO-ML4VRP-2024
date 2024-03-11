import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 
from Load_data import *

max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data = load_data()
def check(route):
    route2 = copy.deepcopy(route)

    # check cap
    cap = 0 
    for x in route2:
        cap += demand[x]
    if cap>max_cap: 
        return False
    
    #check time
    cur_time=0
    for i in range(len(route2)-1):
        cur_time=cur_time+s_time[route2[i]]+distance(route2[i],route2[i+1])
        if cur_time<e_time[route[i+1]]:
            cur_time=e_time[route2[i+1]]
        if cur_time>l_time[route2[i+1]]:
            return False
    return True

def distance(i,j): #tính khoảng cách 2 điểm
    return ((xcoord[i]-xcoord[j])**2+(ycoord[i]-ycoord[j])**2)**(1/2)

def cost2(route):  # tính tổng đường đi của 1 cá thể
    if route[0]!=-1:
        sum=0
        for i in route:
            for j in range(0,len(i)-1):
                sum+=distance(int(i[j]),int(i[j+1]))
        return sum
    else:
        return float('inf')
    
# (-1)
def route_1(routes):
    route=copy.deepcopy(routes)
    for i in range(len(route)):
        for j in range(len(route[i])):
            route[i][j]-=1
    return route



# (+1)
def route__1(routes):
    route=copy.deepcopy(routes)
    for i in range(len(route)):
        for j in range(len(route[i])):
            route[i][j]+=1
    return route




def search(route):
    route=route_1(route)
    for i in range(len(route)-1):
        for j in range(i+1,len(route)):
            for k in range(1,len(route[i])-1):
                for t in range(1,len(route[j])-1):
                        new_route=copy.deepcopy(route)
                        z=new_route[i][k]
                        new_route[i][k]=new_route[j][t]
                        new_route[j][t]=z
                        if check(new_route[i]) and check(new_route[j]) and cost2(new_route)< cost2(route):
                            z=route[i][k]
                            route[i][k]=route[j][t]
                            route[j][t]=z
    return route__1(route)

def search2(routes):
    routes=route_1(routes)
    while routes[-1]==[0,0]:
        routes.pop()
    for i in range(len(routes)):
        for j in range(len(routes)):
            if i!=j:
                k=1
                while (k<len(routes[i])-1):
                    for t in range(1,len(routes[j])-1):
                      if k<len(routes[i])-1:
                        new_route=copy.deepcopy(routes)
                        z=new_route[i][k]
                        new_route[i].pop(k)
                        new_route[j].insert(t,z)
                        if cost2(new_route)< cost2(routes) and check(new_route[j]):
                            routes[i].pop(k)
                            routes[j].insert(t,z)
                        
                    k+=1
    return route__1(routes)


def search4(route, colony, n_customer):
    lst = []
    cus = []
    for key, value in enumerate(route):
        distance = 0
        for i in range(len(value)-1):
            distance += colony.distance_matrix[value[i], value[i+1]]
        lst.append(distance)
    route = route_1(route)

    r = np.random.randint(1,6)
    if r < 3:
        # print(1)
        if np.random.random() < 0.5:
            select_1, select_2, select_3 = np.argsort(np.array(lst))[-3:]
            selected_route = [copy.deepcopy(route[select_1]) ,
                      copy.deepcopy(route[select_2]),
                      copy.deepcopy(route[select_3])]
            a = [select_1, select_2, select_3]
        else:
            select_1, select_2, select_3 = np.random.choice(np.arange(0, len(route)), size=3, replace=False)
            selected_route = [copy.deepcopy(route[select_1]) ,
                      copy.deepcopy(route[select_2]),
                      copy.deepcopy(route[select_3])]
            a = [select_1, select_2, select_3]


    elif r >= 3 and r<4:
        # print(2)
        select_1, select_2, select_3 = np.argsort(np.array([len(value) for value in route]))[:3]
        selected_route = [copy.deepcopy(route[select_1]) ,
                      copy.deepcopy(route[select_2]),
                      copy.deepcopy(route[select_3])]
        a = [select_1, select_2, select_3]

    else:
        # print(3)
        select_1, select_2 = central(route, colony)
        selected_route = [copy.deepcopy(route[select_1]) ,
                      copy.deepcopy(route[select_2])
                      ]
        a = [select_1, select_2]


    if r<4 and len(route[select_1]) + len(route[select_2]) + len(route[select_3]) > n_customer/2:
        select_1, select_2 = central(route, colony)
        selected_route = [copy.deepcopy(route[select_1]) ,
                      copy.deepcopy(route[select_2])
                      ]
        a = [select_1, select_2]

    for i in range(len(selected_route)-1): # Bắt đầu từ route_1
        for j in range(i+1,len(selected_route)): # Lặp qua các route tiếp theo
            for k1 in range(1,len(selected_route[i])-2): # Lặp qua các tp ở route 1
              for k2 in range(k1+1,len(selected_route[i])-1): # 
                for t1 in range(1,len(selected_route[j])-2):
                  for t2 in range(t1+1,len(selected_route[j])-1):
                        new_route=copy.deepcopy(selected_route)
                        zk=copy.deepcopy(new_route[i][k1:k2+1]) # Ok
                        zt=copy.deepcopy(new_route[j][t1:t2+1]) # Ok
                        del new_route[i][k1:k2+1] # Xoá
                        del new_route[j][t1:t2+1] # Xoá 
                        new_route[i]=new_route[i][:k1]+zt+new_route[i][k1:]
                        new_route[j]=new_route[j][:t1]+zk+new_route[j][t1:]
                        if cost2(new_route)< cost2(selected_route) and check(new_route[i]) and check(new_route[j]):
                            zk=copy.deepcopy(selected_route[i][k1:k2+1])
                            zt=copy.deepcopy(selected_route[j][t1:t2+1])
                            del selected_route[i][k1:k2+1]
                            del selected_route[j][t1:t2+1]
                            selected_route[i]=selected_route[i][:k1]+zt+selected_route[i][k1:]
                            selected_route[j]=selected_route[j][:t1]+zk+selected_route[j][t1:]
                            for key, value in enumerate(selected_route):
                                route[a[key]] = value 
                            return route__1(route)
                            
    return route__1(route)




def local_search(t, colony, n_customer):
    t1=copy.deepcopy(t)
    routes=[]
    # routes = t1
    for i in range(len(t1)):
        routes.append(t1[i])
    
    routes=search4(search2(search(routes)), colony, n_customer)
    
    index=0
    result={}
    for x in (routes):
        if x!=[1,1]:
            result[index]=x
            index+=1
    return cost2((route_1(routes))), result