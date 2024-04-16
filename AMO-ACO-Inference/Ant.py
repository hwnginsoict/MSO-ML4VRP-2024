import matplotlib.pyplot as plt
import numpy as np
import math
import random
import copy
from scipy.spatial.distance import cdist


class Ant:
    def __init__(self,data,capacity,q0, heuristic_measure):
        self.data=data
        self.heuristic_measure = heuristic_measure
        self.travel=()
        self.capacity=capacity
        self.time_window={}
        self.pheromon={}
        self.current_point=1
        self.q0=q0
        self.cordination=[]
        self.distance_matrix={}
        self.next_node=1
        self.intensity={}
        self.time_window={}
        self.alpha=1
        self.beta=5
        self.gama=1
        self.theta = 0.75
        self.visited_list=[1]
        self.candidate_list=[]
        self.probability_q0={}
        self.probability_q={}
        self.probability_q_norm={}
        self.minimum_capacity=0
        self.capcities={}
        self.travel_distance=0
        self.rho=0.6
        self.pheromon_numbers={}
        self.Q=1
        self.service_time=0.00
        self.serv_list=[]
        self.f = 2
        self.g = 2

    def customer_cord(self):
        for i in range(len(self.data)):
            cords=[float(self.data[i][1]),float(self.data[i][2])]
            self.cordination.append(cords)
        return self.cordination
    
    def euclidean_distance(self):
        self.heuristic = self.heuristic_measure.detach().numpy()
        for i in range(len(self.cordination)):
            for j in range(len(self.cordination)):
                distance=math.sqrt(((self.cordination[i][0]-self.cordination[j][0])**2)+((self.cordination[i][1]-self.cordination[j][1])**2))
                self.distance_matrix[i+1,j+1]=distance
                try:
                    self.intensity[i+1,j+1]= self.heuristic[i][j]
                except:
                    self.intensity[i+1,j+1]=-99999999
        for i in range(len(self.cordination)):
            self.intensity[i+1,i+1]=-99999999
        return self.distance_matrix,self.intensity
    

    def width_window(self):    
        for i in self.data:
            self.time_window[i[0]]=float(i[5])-float(i[4])
        return self.time_window
    
    def path_pheromon(self):
        for node_i in self.data:
            for node_j in self.data:
                self.pheromon[int(node_i[0]),int(node_j[0])]=1
        return self.pheromon
    

    def make_candidate_list(self):
        self.candidate_list=[]
        for node in self.data:
            if int(node[0]) not in self.visited_list:
                self.candidate_list.append(int(node[0]))
        return self.candidate_list
    


    def choose_next_node(self):
        if len(self.candidate_list)==0:
            self.next_node=1
            return self.next_node
        elif len(self.candidate_list)==1:
            
            self.next_node=self.candidate_list[0]
            if float(self.data[int(self.next_node) - 1][3])<self.capacity and self.service_time + float(self.distance_matrix[self.current_point, self.next_node]) <= float(self.data[self.next_node - 1][5]):
                
                return self.next_node
            else:
                self.next_node=1
                return self.next_node
                
        else:
            next_node=0
            self.probability_q0={}
            self.probability_q={}
            self.probability_q_norm={}
            for node in self.candidate_list:
                w = 1
                if self.service_time + float(self.distance_matrix[self.current_point,node]) < float(self.data[node - 1][4]):
                    w = float(self.data[node - 1][4]) - (self.service_time + float(self.distance_matrix[self.current_point,node]))

                saving = float(self.distance_matrix[self.current_point,1]) + float(self.distance_matrix[1,node]) - self.g * float(self.distance_matrix[self.current_point,node]) + self.f * np.abs(float(self.distance_matrix[self.current_point,1]) - float(self.distance_matrix[1,node]))


                self.probability_q0[self.current_point,node]=(self.pheromon[self.current_point,node]**self.alpha)*(self.intensity[self.current_point,node]**self.beta)*((saving**self.gama)) * ((1/w)**self.theta)
            for node in self.candidate_list:
                w = 1
                if self.service_time + float(self.distance_matrix[self.current_point,node]) < float(self.data[node - 1][4]):
                    w = float(self.data[node - 1][4]) - (self.service_time + float(self.distance_matrix[self.current_point,node]))
                saving = float(self.distance_matrix[self.current_point,1]) + float(self.distance_matrix[1,node]) - self.g * float(self.distance_matrix[self.current_point,node]) + self.f * np.abs(float(self.distance_matrix[self.current_point,1]) - float(self.distance_matrix[1,node]))

                self.probability_q[self.current_point,node]=(self.pheromon[self.current_point,node]**self.alpha)*(self.intensity[self.current_point,node]**self.beta)*((saving**self.gama)) * ((1/w)**self.theta)/ max(self.probability_q0.values())

            def softmax_normalize(dictionary):
                values = np.array(list(dictionary.values()), dtype=np.float64)
                exp_values = np.exp(values - np.max(values)) 
                normalized_values = exp_values / np.sum(exp_values)
                normalized_dict = dict(zip(dictionary.keys(), normalized_values))
                return normalized_dict
            
            
            self.probability_q_norm =softmax_normalize(self.probability_q)
            self.capcities={}
            for node in self.candidate_list:
                self.capcities[node]=float(self.data[node-1][3])
            q=random.random()
            self.next_node = None

            if q<=self.q0:

                sorted_value_q0=sorted(self.probability_q0.values(),reverse=True)
                for i in range(len(sorted_value_q0)):
                    for key,value in self.probability_q0.items():
                        if value==sorted_value_q0[i]:
                            if float(self.data[key[1]-1][3])<=self.capacity and self.service_time+ float(self.distance_matrix[key[1], key[0]]) <=float(self.data[key[1]-1][5]) :
                                next_node=key[1]
                                self.next_node=next_node
                                return self.next_node
  
            else:
                def roulette_wheel_selection(values, probabilities):
                    selected_key = random.choices(list(values), weights=list(probabilities), k=1)[0]
                    return selected_key
                for item in self.probability_q_norm:
                    selected_key = roulette_wheel_selection(self.probability_q_norm.keys(), self.probability_q_norm.values())
                    if float(self.data[selected_key[1]-1][3])<=self.capacity and self.service_time+ float(self.distance_matrix[selected_key[1], selected_key[0]])<=float(self.data[selected_key[1] - 1][5]):
                        next_node=selected_key[1]
                        self.next_node=next_node
                        return self.next_node
                    else:
                        continue
                self.next_node=None
                return self.next_node
            return self.next_node
            
    
    def move(self):
        if self.next_node==None:
            self.next_node=1
            self.travel=(self.current_point,1)
        else:
            self.visited_list.append(self.next_node)
            self.travel=(self.current_point,self.next_node)
            if self.service_time + self.distance_matrix[self.travel[0], self.travel[1]] < float(self.data[self.travel[1]-1][4]):
                self.service_time=float(self.data[self.travel[1]-1][4])+float(self.data[self.travel[1]-1][6])
            else:
                self.service_time += float(self.distance_matrix[self.travel[0], self.travel[1]]) + float(self.data[self.travel[1]-1][6])

            self.serv_list.append(self.service_time)
            self.capacity=self.capacity-float(self.data[self.next_node-1][3])
            self.current_point=self.next_node
        
        self.travel_distance+=self.distance_matrix[self.travel]

        return self.travel
    
    def update_rho(self):
        self.rho=0.9*self.rho
        return self.rho
    
    

    def update_pheromon(self,ants_travels, distance):
        for travel in ants_travels:
            self.pheromon[travel] = self.pheromon[travel] * (1-self.rho) + 1/distance
            
        return self.pheromon
    
    def update_global(self, ants_travels, distance):
        for travel in ants_travels:
            self.pheromon[travel] += 1/distance
        return self.pheromon
    
    
    def update_BTNT(self, ants_travels, distance, alpha, pop_size):
        for travel in ants_travels:
            self.pheromon[travel] += 2*(alpha)*pop_size/distance
        return self.pheromon