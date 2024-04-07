import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist
from ultis import *
from Ant import Ant 

def load_data():
    xcoord=np.array([])
    ycoord=np.array([])
    demand=np.array([])
    e_time=np.array([])
    l_time=np.array([])
    s_time=np.array([])

    data = open('Data.txt','r')
    lines = data.readlines()
    for i in range(len(lines)):
        if lines[i]=='NUMBER     CAPACITY\n':
            veh_num,max_cap=map(int,lines[i+1].strip().split())
        if lines[i]=='CUSTOMER\n':
            j=i+3
            while j<len(lines):
                a,b,c,d,e,f,g=map(int,lines[j].strip().split())
                xcoord=np.append(xcoord,b)
                ycoord=np.append(ycoord,c)
                demand=np.append(demand,d)
                e_time=np.append(e_time,e)
                l_time=np.append(l_time,f)
                s_time=np.append(s_time,g)
                j+=1
    cus_num=len(demand)-1

    data=[]
    for i in range(1,cus_num+2):
        new_data=[str(i)]
        new_data.append(str(int(xcoord[i-1])))
        new_data.append(str(int(ycoord[i-1])))
        new_data.append(str(int(demand[i-1])))
        new_data.append(str(int(e_time[i-1])))
        new_data.append(str(int(l_time[i-1])))
        new_data.append(str(int(s_time[i-1])))
        data.append(new_data)

    return max_cap, xcoord, ycoord, demand, e_time, l_time, s_time, data
