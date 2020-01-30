import auxiliary_tools
import random
import requests
import numpy as np
from tqdm import tqdm 
import time
import matplotlib.pyplot as plt
import matplotlib
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from geopy.distance import geodesic


def crossover(individual_a, individual_b,crossover_probability):
    child_a = []
    child_b = []

    for i, (gene_a, gene_b) in enumerate(zip(individual_a, individual_b)):
        if random.random() < crossover_probability:
            child_a.append(gene_b)
            child_b.append(gene_a)
        else:
            child_a.append(gene_a)
            child_b.append(gene_b)

    return np.array(child_a), np.array(child_b)

def mutation(individual, mutation_probability, rand_dist_min, rand_dist_max):
    mutated_individual = []
    for gene in individual:
        if random.random() < mutation_probability:
            rand_distance = random.randint(rand_dist_min, rand_dist_max)/1000
            rand_angle = random.randint(1, 360)
            
            mutated_gene = geodesic(kilometers=rand_distance).destination(gene, rand_angle)[:2]
            mutated_individual.append( mutated_gene )
        else:
            mutated_individual.append( gene )
    return  np.array(mutated_individual)

def client_fitness(route_requests, individual):
    c_fitness = []
    for i in range(len(route_requests)):
        request_r = route_requests[i]
        request_origin = [request_r[3], request_r[2]]
        vs_individual = individual[i]
        vs_destination = vs_individual
        c_fitness.append(auxiliary_tools.getGeoDistanceETA_OSRM(request_origin, vs_destination, 5001, 'walking'))
    fitness_value = np.sum([f[0] for f in c_fitness])
    return fitness_value, c_fitness

def operator_fitness(individual, penalty_const):
    ori_dest = [(first, second) for first, second in zip(individual, individual[1:])]
    penalty_sum = 0
    for pair in ori_dest:
        if max(pair[0] != pair[1]) == True:
            penalty_sum+=penalty_const
    o_fitness = []
    for od_r in ori_dest:
        o_fitness.append(auxiliary_tools.getGeoDistanceETA_OSRM(od_r[0], od_r[1], 5000, 'driving'))
        
    fitness_value = np.sum([f[0] for f in o_fitness]) + penalty_sum
    return fitness_value, o_fitness

def createIndividual(route_requests, search_data):
    individual = []
    for request in route_requests:
        bag_genes = search_data[int(request[1])]
        gene = np.array(random.choice(bag_genes))
        individual.append(gene)
    individual = np.array(individual)
    return individual

def plot_solution(request, individual, image_path, min_longitude, max_longitude, min_latitude, max_latitude):
    
    df_individual = pd.DataFrame(individual).rename(columns={0:'latitude', 1:'longitude'})
    df_request = pd.DataFrame(request).rename(columns={3:'latitude', 2:'longitude'})
    
    BBox = ((min_longitude, max_longitude, min_latitude, max_latitude))
    gyn_m = plt.imread(image_path)
    fig, ax = plt.subplots(figsize = (10,10))

    individual_longitude = df_individual.longitude
    individual_latitude = df_individual.latitude
    ax.scatter(individual_longitude, individual_latitude, zorder=1, alpha= 1, c='r', s=5)

    request_longitude = df_request[['longitude']].to_numpy()
    request_latitude = df_request[['latitude']].to_numpy()
    BBox = ((request_longitude.min(), request_longitude.max(), request_latitude.min(), request_latitude.max()))       

    ax.set_title('Indivual Solution')
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    
    ax.scatter(request_longitude, request_latitude, zorder=1, alpha= 1, c='b', s=5)

    for i in range(0, len(df_individual), 1):
        ax.plot([request_longitude[i], individual_longitude[i]], [request_latitude[i], individual_latitude[i]], 'k-', alpha=1, linewidth=0.7)

    ax.imshow(gyn_m, alpha= 0.6, zorder=0, extent = BBox, aspect='auto')

    # inset axes....
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.imshow(gyn_m, extent=BBox, interpolation="nearest", origin="upper")

    axins.scatter(individual_longitude, individual_latitude, zorder=1, alpha= 1, c='r', s=15)
    axins.scatter(request_longitude, request_latitude, zorder=1, alpha= 1, c='b', s=15)
    for i in range(0, len(df_individual), 1):
        axins.plot([request_longitude[i], individual_longitude[i]], [request_latitude[i], individual_latitude[i]], 'k-', alpha=1, linewidth=0.7)

    x1, x2, y1, y2 = -49.28, -49.27, -16.715, -16.705
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    ax.indicate_inset_zoom(axins)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='Request',
                              markerfacecolor='b', markersize=10),
                       Line2D([0], [0], marker='o', color='w', label='Virtual Point',
                              markerfacecolor='r', markersize=10),
                       Line2D([0], [0], color='k', lw=4, label='Walking')]

    ax.legend(handles=legend_elements, loc="upper left")
    plt.axis("off")
    plt.show()
    
def plot_points(df_points, image_path, min_longitude, max_longitude, min_latitude, max_latitude, title):
    
    df_request = df_points
    BBox = ((min_longitude, max_longitude, min_latitude, max_latitude))
    gyn_m = plt.imread(image_path)
    fig, ax = plt.subplots(figsize = (10,10))

    request_longitude = df_request[['LONGITUDE']].to_numpy()
    request_latitude = df_request[['LATITUDE']].to_numpy()
    BBox = ((request_longitude.min(), request_longitude.max(), request_latitude.min(), request_latitude.max()))       

    ax.set_title(title)
    ax.set_xlim(BBox[0],BBox[1])
    ax.set_ylim(BBox[2],BBox[3])
    
    ax.scatter(request_longitude, request_latitude, zorder=1, alpha= 1, c='b', s=5)

    ax.imshow(gyn_m, alpha= 0.6, zorder=0, extent = BBox, aspect='auto')

    # inset axes....
    axins = ax.inset_axes([0.5, 0.5, 0.47, 0.47])
    axins.imshow(gyn_m, extent=BBox, interpolation="nearest", origin="upper")

    axins.scatter(request_longitude, request_latitude, zorder=1, alpha= 1, c='b', s=15)
    
    x1, x2, y1, y2 = -49.28, -49.27, -16.715, -16.705
    axins.set_xlim(x1, x2)
    axins.set_ylim(y1, y2)
    axins.set_xticklabels('')
    axins.set_yticklabels('')

    ax.indicate_inset_zoom(axins)

    legend_elements = [Line2D([0], [0], marker='o', color='w', label='client request',
                              markerfacecolor='b', markersize=10)]

    ax.legend(handles=legend_elements, loc="upper left")
    plt.axis("off")
    plt.show()