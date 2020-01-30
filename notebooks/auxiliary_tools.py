import os
import time
import json
import fiona
import geopy
import shutil
import pickle
import random
import requests
import matplotlib
import numpy as np
import kml2geojson
import pandas as pd
from tqdm import tqdm
import geopy.distance
import geopandas as gpd
from shapely import geometry
from datetime import datetime
import matplotlib.path as mplPath
from geopy.distance import geodesic
from datetime import datetime, timedelta
from shapely.geometry import Point, Polygon

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
from geopy.distance import great_circle
from shapely.geometry import MultiPoint

pd.options.display.max_columns = None
fiona.supported_drivers['KML'] = 'rw'
fiona.supported_drivers['LIBKML'] = 'rw'

GOOGLE_API_KEY = "__"

def getPenaltyConst(pic_drop_time):
    df = pd.read_csv("../data/requests_table.csv", sep='\t', encoding='utf-16', error_bad_lines=False)
    df['KM_PER_HOUR'] = df['KM_PER_HOUR'].str.replace(",", ".").astype(float)
    avg_km_hour = df['KM_PER_HOUR'].mean()
    return (avg_km_hour*pic_drop_time)/60

def getGeoDistanceETA_GMAPS(origin, destination, transport_mode):
    r = requests.get(url = "https://maps.googleapis.com/maps/api/distancematrix/json", 
                                 params = {"units": "metric",
                                            "origins":str(origin[0])+","+str(origin[1]),
                                            "destinations": str(destination[0])+","+str(destination[1]),
                                            "mode": transport_mode,
                                            "key": GOOGLE_API_KEY,})
    data = r.json()
    # print(data)
    if data["rows"][0]["elements"][0]['status'] != "NOT_FOUND":
        distance = data["rows"][0]['elements'][0]['distance']['value'] / 1000
        eta =  round(data["rows"][0]['elements'][0]['duration']['value'] / 60,3)
    else:
        distance = np.nan
        eta = np.nan
    return distance, eta

def getGeoDistanceETA_OSRM(origin, destination, server_port, transport_mode):
    
    parameter = ",".join([str(origin[1]), str(origin[0])])+";"+ ",".join([str(destination[1]), str(destination[0])])
    page = ''
    while page == '':
        try:
            page = requests.get(url = "http://127.0.0.1:"+str(server_port)+"/route/v1/"+transport_mode+"/"+parameter)
            break
        except:
            time.sleep(2)
            continue
    data = page.json()
    distance = data["routes"][0]["distance"]/1000
    duration = data["routes"][0]["duration"]/60

    return distance, duration

def client_fitness(route_requests, individual):
    c_fitness = []
    for i in tqdm(range(len(route_requests))):
        request_r = route_requests[i]
        request_origin = [request_r[3], request_r[2]]
        vs_individual = individual[i]
        vs_destination = vs_individual
        c_fitness.append(getGeoDistanceETA_OSRM(request_origin, vs_destination, 5000, 'walking'))
    fitness_value = np.sum([f[0] for f in c_fitness])
    return fitness_value, c_fitness

def genFilesIDS():
    # Requests
    n = 300
    chunk_requests_ids = [requests_ids[x:x+n] for x in range(0, len(requests_ids), n)]
    for chunk in chunk_requests_ids:
        idx = chunk_requests_ids.index(chunk)
        with open('./request_ids/requests_ids_'+str(idx)+'.txt', 'w') as f:
            for item in chunk:
                f.write("%s, " % item)
    
    # Proposals
    len(range(441358, 564760)) - len(set(df_proposals.ID))
    missing_ids = list(set(range(441358, 564760)) - set(df_proposals.ID))
    n = 300
    chunk_missing_ids = [missing_ids[x:x+n] for x in range(0, len(missing_ids), n)]
    with open('missing_ids_3.txt', 'w') as f:
        for item in chunk_missing_ids[3]:
            f.write("%s, " % item)
    
    # RIDES
    n = 300
    ride_ids = list(range(136985,183601))
    chunk_rides_ids = [ride_ids[x:x+n] for x in range(0, len(ride_ids), n)]
    for chunk in chunk_rides_ids:
        idx = chunk_rides_ids.index(chunk)
        with open('./ride_ids/ride_ids_'+str(idx)+'.txt', 'w') as f:
            for item in chunk:
                f.write("%s, " % item)

def prepProposals(path):
    proposal_files = [file for file in os.listdir(path) if ".csv" in file ]
    frames = []
    for file in proposal_files:
        frames.append(pd.read_csv(path+file, sep=',', encoding='UTF-8'))
    df_proposals = pd.concat(frames)
    df_proposals = df_proposals.rename(columns=lambda x: x.upper().replace(" ","_").replace("#","N"))
    df_proposals = df_proposals.drop_duplicates()
 
    requests_ids = list(set(df_proposals['REQUEST_ID']))
    
    df_proposals_shrt = df_proposals[['ID', 'ISSUE', 'RIDER_ID', 'REQUEST_ID',
                'NUMBER_OF_PASSENGERS', 'RIDE_COST','VAN_ID',
                'ACCEPTED','PICKUP_ETA', 'DROPOFF_ETA', 
                'PICKUP_WALK_DURATION', 'PRICING_DISTANCE']]
    df_proposals_shrt = df_proposals_shrt.rename(columns={'ID':'PROPOSAL_ID', 'ISSUE':'PROPOSAL_TIMESTAMP',
                    'NUMBER_OF_PASSENGERS':'N_PASSENGERS',
                    'PICKUP_ETA':'PICKUP_TIMESTAMP', 'DROPOFF_ETA':'DROPOFF_TIMESTAMP', 
                    'PICKUP_WALK_DURATION':'WALKING_DISTANCE_METERS', 'PRICING_DISTANCE':'RIDE_DISTANCE_METERS'})
    
    df_proposals_shrt.replace('None', np.nan, inplace=True)
    df_proposals_shrt['RIDE_DISTANCE_METERS'] = df_proposals_shrt['RIDE_DISTANCE_METERS'].apply(lambda x: int((float(x.strip(' miles'))/0.62137119)*1000) )
    
    
    df_proposals_shrt = df_proposals_shrt.astype({'PROPOSAL_ID': float, 'PROPOSAL_TIMESTAMP': float, 'RIDER_ID': float, 'REQUEST_ID': float,
                          'N_PASSENGERS': float, 'RIDE_COST': float, 'VAN_ID': float, 'ACCEPTED': bool,
                         'PICKUP_TIMESTAMP': float, 'DROPOFF_TIMESTAMP': float, 'WALKING_DISTANCE_METERS': float,
                         'RIDE_DISTANCE_METERS': float})
    print("# proposals: ",len(df_proposals_shrt['PROPOSAL_ID']))
    return df_proposals_shrt, requests_ids

def prepRequests(path, requests_ids):
    header = 'Id,Last Update,Request,Ride Id,Rider Id,Rider Status,# pax,Origin Lat,Origin Lng,Origin Address,Destination Lat,Destination Lng,Destination Address,Err Statement,City\n'
    requests_data = [header]
    requests_files = [file for file in os.listdir(path) if ".csv" in file ]
    for file in requests_files:
        with open(path+file, 'r') as f:
            line = f.readline()
            header = line
            while line:
                line = f.readline()
                tagged_line = line.replace("|","|<dw>").split("|")
                fixed_line = "".join([x.replace(", "," ").strip('<dw>') if '<dw>' in x else x for x in tagged_line])
                requests_data.append(fixed_line)
    path = "./new_requests/"
    if not os.path.exists(path):
        os.makedirs(path)
    with open("./new_requests/processed_requests.csv", 'w') as f:
        for line in requests_data:
            f.write(line)
    requests_files = [file for file in os.listdir(path) if ".csv" in file ]
    frames = []
    for file in requests_files:
        frames.append(pd.read_csv(path+file, sep=',', encoding='UTF-8'))
    df_requests = pd.concat(frames, sort=False)
    df_requests = df_requests.rename(columns=lambda x: x.upper().replace(" ","_").replace("#","N"))
    df_requests = df_requests.drop_duplicates()
    shutil.rmtree(path)
    df_requests_shrt = df_requests[['ID', 'REQUEST','ORIGIN_LAT',
                                'ORIGIN_LNG', 'DESTINATION_LAT', 'DESTINATION_LNG']]

    df_requests_shrt = df_requests_shrt.rename(columns={'ID':'REQUEST_ID', 'REQUEST':'REQUEST_TIMESTAMP'})
    df_requests_shrt.replace('None', np.nan, inplace=True)
    # df_requests_shrt = df_requests_shrt.fillna(0)
    df_requests_shrt = df_requests_shrt.astype({'REQUEST_ID': float, 'REQUEST_TIMESTAMP': float, 'ORIGIN_LAT': float,
                          'ORIGIN_LNG': float, 'DESTINATION_LAT': float, 'DESTINATION_LNG': float })
    print("# requests",len(set(df_requests_shrt['REQUEST_ID'])))
    return df_requests_shrt

def prepRides(path):
    ride_files = [file for file in os.listdir(path) if ".csv" in file ]
    frames = []
    for file in ride_files:
        frames.append(pd.read_csv(path+file, sep=',', encoding='UTF-8'))
    df_rides = pd.concat(frames)
    df_rides = df_rides.rename(columns=lambda x: x.upper().replace(" ","_").replace("#","N"))
    df_rides = df_rides.drop_duplicates()
    df_rides_shrt = df_rides[['ID', 'STATUS', 'RIDER_ID', 'REQUEST_ID', 'PROPOSAL_ID', 'DRIVER_ID', 'ACCEPTED', "CANCELLED"]]
    df_rides_shrt = df_rides_shrt.rename(columns={'ID':"RIDE_ID", 'ACCEPTED':'ACCEPTED_TIMESTAMP', "CANCELLED":"CANCELLED_TIMESTAMP"})
    df_rides_shrt.replace('None', np.nan, inplace=True)
    df_rides_shrt.loc[:, 'STATUS'] = df_rides_shrt.STATUS.apply(lambda x: x.split('<')[0].split(" by")[0].title())
    # df_rides_shrt.loc[:, 'ACCEPTED_TIMESTAMP'] = df_rides_shrt.ACCEPTED_TIMESTAMP.apply(lambda x: int(x))
    # df_rides_shrt.loc[:, 'CANCELLED_TIMESTAMP'] = df_rides_shrt.CANCELLED_TIMESTAMP.apply(lambda x: float(str(x).split(".")[0]))
    df_rides_shrt = df_rides_shrt.astype({'RIDE_ID':float, 'STATUS':str, 'RIDER_ID':float, 'REQUEST_ID':float, 'PROPOSAL_ID':float,
                                      'DRIVER_ID':float, 'ACCEPTED_TIMESTAMP':float, 'CANCELLED_TIMESTAMP':float})
    
    print("# rides",len(set(df_rides_shrt['RIDE_ID'])))
    return df_rides_shrt

def mergePropRequRides(df_proposals_shrt, df_requests_shrt, df_rides_shrt):
    
    df_prop_requ = pd.merge(df_proposals_shrt, df_requests_shrt, on='REQUEST_ID', how='outer')
    df_prop_requ_ride = pd.merge(df_prop_requ, df_rides_shrt, on=['REQUEST_ID', 'RIDER_ID', 'PROPOSAL_ID'], how='outer')
    df_drt_dataset = df_prop_requ_ride[sorted(df_prop_requ_ride.columns)]
    df_drt_dataset = df_drt_dataset.reset_index(drop=True)
    df_drt_dataset.to_csv("../data/drt_dataset.csv",sep="\t", index=False)
    
    return df_drt_dataset


def getRoad(list_coordinates):
    list_coordinates = [[x[1], x[0]] for x in list_coordinates]
    coordinates = "|".join([ ",".join([str(x) for x in y]) for y in list_coordinates])
    r = requests.get(url = "https://roads.googleapis.com/v1/nearestRoads", 
                             params = {"points": coordinates,
                                        "key":GOOGLE_API_KEY})
    data = r.json()
    
    original_idx = sorted([x['originalIndex'] for x in data['snappedPoints']], reverse=True)
    idx = sorted(list(range(len(original_idx))), reverse=True)
    idx_filter = list(dict(zip(original_idx, idx)).values())
    
    roads_coordinates = [[x['location']['longitude'], x['location']['latitude']] for x in data['snappedPoints']]
    filtered_roads = np.array(roads_coordinates)[idx_filter]
    
    return [[x[1], x[0]] for x in filtered_roads]

def getAddress(lat, long):
    r = requests.get(url = "https://maps.googleapis.com/maps/api/geocode/json", 
                             params = {"latlng":str(lat)+","+str(long),
                                        "key":GOOGLE_API_KEY})
    data = r.json()
    
    for r in data['results']:
        return r['formatted_address']
        
    return None

def getRoadsFromCenterPoints(path, open_filename, save_filename):
    with open(path+open_filename+".geojson", "r") as f:
        centers = json.load(f)
    n = 100
    center_coordinates = [x['geometry']['coordinates'][0:2] for x in centers['features']]
    chunk_center_coordinates = [center_coordinates[x:x+n] for x in range(0, len(center_coordinates), n)]
    filtered_roads = []
    for chunk in tqdm(chunk_center_coordinates):
        filtered_roads.extend(getRoad(chunk))
    n_road_points = len(filtered_roads)
    dummy_road_indexes = list(zip([0]*n_road_points,range(n_road_points)))
    road_points = generatePointGeojson(filtered_roads, dummy_road_indexes, save_filename, path, True)
    print("GeoJson2KML CENTER")
    convertGeoJson2KML(path+save_filename+".geojson", path+save_filename+".kml")

def loadVirtualStops(virtual_stop_path):
    with open(virtual_stop_path, "r") as f:
        centers = json.load(f)
    vs_coordinates = [np.array(x['geometry']['coordinates'][0:2]) for x in centers['features']]
    return np.array(vs_coordinates)

def loadRidesData(rides_path):
    data = pickle.load(open(rides_path, "rb"))
    Y = []
    for van, route in data.items():
        route_data = []
        for route_id, value in route.items():
            route_data.append([van , route_id, value[0][0], value[0][1], int(value[2].replace('P','0').replace('D','1'))])
        Y.append(np.array(route_data))
    Y = np.array(Y)
    return Y, data

def generatePointGeojson(in_area, in_area_indexes, filename, path, flag_write):
    dict_citybus_grid = {'type':'FeatureCollection', 'features':[], 'name':filename}
    i = 1
    for c in range(0,len(in_area)):
        fea = {'type': 'Feature', 'properties': {'name': filename.lower()+'_'+str(in_area_indexes[c][0]+1)+'_'+str(in_area_indexes[c][1]+1)},
              'geometry': {'type': 'Point', 'coordinates': [in_area[c][1], in_area[c][0], 0] } }
        i+=1
        dict_citybus_grid['features'].append(fea)
    
    if flag_write:
        with open(path+filename+".geojson", 'w', encoding='utf8') as f:
            json.dump(dict_citybus_grid, f)
    
    return dict_citybus_grid

def generatePolygonGeojson(area, area_indexes, filename, path, flag_write):
    invert_area = [[[j[1], j[0]] for j in i] for i in area]
    dict_citybus_grid = {'type':'FeatureCollection', 'features':[], 'name':filename}
    i = 1
    for c in range(0,len(invert_area)):
        fea = {'type': 'Feature', 'properties': {'name': filename.lower()+'_'+str(area_indexes[c][0]+1)+'_'+str(area_indexes[c][1]+1)},
              'geometry': {'type': 'Polygon', 'coordinates': [invert_area[c]] } }
        i+=1
        dict_citybus_grid['features'].append(fea)
    if flag_write:
        with open(path+filename+".geojson", 'w', encoding='utf8') as f:
            json.dump(dict_citybus_grid, f)
  
    return dict_citybus_grid

def get_random_color(pastel_factor = 0.5):
    return [(x+pastel_factor)/(1.0+pastel_factor) for x in [random.uniform(0,1.0) for i in [1,2,3]]]

def color_distance(c1,c2):
    return sum([abs(x[0]-x[1]) for x in zip(c1,c2)])

def generate_new_color(existing_colors,pastel_factor = 0.5):
    max_distance = None
    best_color = None
    for i in range(0,100):
        color = get_random_color(pastel_factor = pastel_factor)
        if not existing_colors:
            return color
        best_distance = min([color_distance(color,c) for c in existing_colors])
        if not max_distance or best_distance > max_distance:
            max_distance = best_distance
            best_color = color
    return best_color

def gen_list_distinct_colors(n):
    colors = []

    for i in range(0,n):
        colors.append(generate_new_color(colors,pastel_factor = 0.9))

    colors_hex = []
    for i in colors:
        colors_hex.append(matplotlib.colors.to_hex(i).strip('#'))
    return colors_hex

def construct_kml_list(dict_routes, label):
    import simplekml
    kml = simplekml.Kml()
    n = len(dict_routes.keys())
    hex_colors = gen_list_distinct_colors(n)
    dict_colors = dict(zip(set(dict_routes.keys()), hex_colors))
    for route_id, routes_values in dict_routes.items():
        fol = kml.newfolder(name='ROUTE_'+str(route_id))
        for ride_id, tuple_loc in routes_values.items():
            location = tuple_loc[0]
            time = tuple_loc[1]
            tipo = tuple_loc[2]
            client = tuple_loc[3]
            pnt = fol.newpoint(name=str(ride_id)+"_"+str(client)+"_"+tipo, coords=[location])
            pnt.style.iconstyle.color = simplekml.Color.hex(dict_colors[route_id])
            pnt.style.iconstyle.icon.href = 'http://www.gstatic.com/mapspro/images/stock/503-wht-blank_maps.png'
            
        ls = fol.newlinestring(name=str(route_id)+"_"+str('VAN'))
        ls.coords = [ x[0] for x in list(dict_routes[route_id].values())]
        ls.style.linestyle.color = simplekml.Color.hex(dict_colors[route_id])  # Red
        ls.style.linestyle.width = 5  # 10 pixels
    
    kml.save(label, format=True)

def convertKMLPolygon2Info(filename, path):
    kml2geojson.main.convert(path+filename+".kml", path)
    with open(path+filename+".geojson", "r") as f:
        region_geojson = f.readlines()[0]
        region = json.loads(region_geojson)
    list_polygon = []
    names = []
    region_description = []
    for i in tqdm(range(0,len(region["features"]))):
        region_name = region["features"][i]['properties']['name']
        try:
            region_description.append( region["features"][i]['properties']['description'])
        except:
             region_description.append(None)
        names.append(region_name)
        region_coordinates = region["features"][i]['geometry']['coordinates']
        polygon = region_coordinates[0]
        x_y_polygon = [[coord[0], coord[1]] for coord in polygon]
        list_polygon.append(x_y_polygon)

    return list_polygon, names, region_description, region

def convertGeoJson2KML(geojson_path, kml_path):
    data = gpd.read_file(geojson_path, driver="GeoJSON")
    data.to_file(kml_path, driver="KML")

def extendPolygonData(list_polygon):
    citybus_polygon_ext = []
    for p in list_polygon:
        citybus_polygon_ext.extend(p)
    return citybus_polygon_ext

def createGrid(citybus_polygon_ext,distance):
    min_long = np.array(citybus_polygon_ext)[:,0].min()
    max_long = np.array(citybus_polygon_ext)[:,0].max()+0.1
    min_lat = np.array(citybus_polygon_ext)[:,1].min()
    max_lat = np.array(citybus_polygon_ext)[:,1].max()+0.1

    origin = geopy.Point(max_lat, min_long)
    origin_iter = [max_lat, min_long]

    d = distance

    grid = []
    i = 0
    column_point = origin
    column_iter = origin_iter
    while(column_iter[0] >= min_lat):
        fst_column_iter = [geodesic(kilometers=d).destination(column_point, 180).latitude, geodesic(kilometers=d).destination(column_point, 180).longitude]
        fst_column_point =  geopy.Point(fst_column_iter[0], fst_column_iter[1])
        j = 0
        grid_row = []
        while (column_iter[1] <= max_long):
            cell_r = [geodesic(kilometers=d).destination(column_point, 90).latitude, geodesic(kilometers=d).destination(column_point, 90).longitude]
            grid_row.append(column_iter)
            column_iter = cell_r
            column_point = geopy.Point(cell_r[0], cell_r[1])
            j+=1
        grid.append(grid_row)
        column_iter = fst_column_iter
        column_point = fst_column_point
        i+=1
        max_row = i
        max_column = j

    return grid, max_row, max_column

def filterPointsPolygon(grid_centers, d ,citybus_polygon, grid_indexes):
    count = 0
    in_area = []
    in_area_indexes = []
    for c in tqdm(range(0,len(grid_centers))):
        for r in range(0,len(citybus_polygon)):
            polygon_path = mplPath.Path(citybus_polygon[r])
            center = tuple([grid_centers[c][1], grid_centers[c][0]])
            aux = geodesic(kilometers=d/2).destination(center, -180)
            left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=d/2).destination(center, 0)
            right = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(center, 45)
            c_u_left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(center, 135)
            c_u_right = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(center, 225)
            c_d_left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(center, 315)
            c_d_right = [aux.latitude, aux.longitude]
            if polygon_path.contains_point(center) or polygon_path.contains_point(c_u_left) or polygon_path.contains_point(c_u_right) or polygon_path.contains_point(c_d_left) or polygon_path.contains_point(c_d_right):
                in_area.append(tuple([grid_centers[c][0], grid_centers[c][1]]))
                in_area_indexes.append(grid_indexes[c])
                break
    return in_area, in_area_indexes

def createGridCenters(grid, max_row, max_column):
    i = 0
    grid_centers = []
    grid_indexes = []
    for i in tqdm(range(0,max_row-1)):
        row_center = []
        for j in range(0,max_column-1):
            poly = geometry.Polygon([grid[i][j], grid[i][j+1], grid[i+1][j+1], grid[i+1][j]])
            row_center.append([poly.centroid.x, poly.centroid.y])
            grid_indexes.append((i,j))
        grid_centers.extend(row_center)   
    return grid_centers, grid_indexes

def extendGrid(grid, max_row, max_column):
    grid_cells = []
    for i in tqdm(range(0,max_row-1)):
        for j in range(0, max_column-1):
            c = grid[i][j]
            r = grid[i][j+1]
            d = grid[i+1][j+1]
            l = grid[i+1][j]
            cell = [c,r,d,l,c]
            grid_cells.append(cell)
    return grid_cells

def filterGridPolygon(grid_cells, grid_indexes_filtered, max_row, max_column):
    grid_cells_matrix = []
    k = 0
    for i in tqdm(range(0,max_row-1)):
        grid_cells_row = []
        for j in range(0,max_column-1):
            grid_cells_row.append(grid_cells[k])
            k+=1
        grid_cells_matrix.append(grid_cells_row)

    grid_cells_filtered = []
    for c in grid_indexes_filtered:
        grid_cells_filtered.append(grid_cells_matrix[c[0]][c[1]])
    return grid_cells_filtered

def createGeojsonPolygon(polygon_filename, distance, filename, path):
    polygons, names, descriptions, geojson = convertKMLPolygon2Info(polygon_filename, path)
    list_coordinates = extendPolygonData(polygons)
    print("Gen GRID")
    grid, max_row, max_column = createGrid(list_coordinates, distance)
    grid_centers, grid_indexes = createGridCenters(grid, max_row, max_column)
    in_grid_centers, in_grid_indexes = filterPointsPolygon(grid_centers, distance, polygons, grid_indexes)
    grid_extended = extendGrid(grid, max_row, max_column)
    in_grid_extended = filterGridPolygon(grid_extended, in_grid_indexes, max_row, max_column)

#     print("Extend Center with Corners")
#     grid_corners = []
#     for x in in_grid_extended:
#         for y in x:
#             grid_corners.append(tuple(y))

#     in_grid_centers.extend(grid_corners)
#     n_points = len(in_grid_centers)
#     dummy_indexes = list(zip([0]*n_points,range(n_points)))

    print("Gen Geojson CENTER")
    in_grid_centers_geojson = generatePointGeojson(in_grid_centers, in_grid_indexes, "CENTER", path, True)
    print("Gen Sample Geojson Center")
    in_grid_centers_sample = list(random.sample(in_grid_centers, k=2000))
    n_sample_points = len(in_grid_centers_sample)
    dummy_indexes_sample = list(zip([0]*n_sample_points,range(n_sample_points)))
    in_grid_centers_geojson = generatePointGeojson(in_grid_centers_sample, dummy_indexes_sample, "SAMPLE_CENTER", path, True)
    print("GeoJson2KML CENTER")
    convertGeoJson2KML(path+"CENTER.geojson", path+"CENTER.kml")
    print("GeoJson2KML CENTER")
    convertGeoJson2KML(path+"SAMPLE_CENTER.geojson", path+"SAMPLE_CENTER.kml")
    
def createRoutesWithVS(path, distance, Y, dict_routes, VS):
    new_VS = []
    for i in tqdm(range(len(Y))):
        route_geojsons = []
        new_vs_route = []
        for y in Y[i]:
            lat = y[2:4][1]
            long = y[2:4][0]
            label = y[1]

            center = geopy.Point(lat, long)
            d = distance
            c_u_right = tuple(geodesic(kilometers=d).destination(center, 45))
            c_d_right = tuple(geodesic(kilometers=d).destination(center, 135))
            c_d_left = tuple(geodesic(kilometers=d).destination(center, 225))
            c_u_left = tuple(geodesic(kilometers=d).destination(center, 315))
            coords = [c_u_right, c_d_right, c_d_left, c_u_left]
            poly = Polygon(coords)

            vs_in_area = []
            for stop in VS:
                p = Point(stop[1], stop[0])
                if p.within(poly):
                    vs_in_area.append((stop[1],stop[0]))

            new_vs_route.append(np.array(vs_in_area))

            dummy_indexes = list(zip([label]*len(vs_in_area),range(len(vs_in_area))))

            dummy_area_idx = list(zip([label]*len([coords]),range(len([coords]))))

            vs_in_area_geojson = generatePointGeojson(vs_in_area, dummy_indexes, "ROUTE_VIRTUAL_POINTS", path, False)
            point_area_geojson = generatePolygonGeojson([coords], dummy_area_idx, "AREA_VIRTUAL_POINTS", path, False)

            vs_in_area_geojson['features'].extend(point_area_geojson['features'])
            route_geojsons.append(vs_in_area_geojson)

        new_VS.append(np.array(new_vs_route))
        construct_kml_list({i : dict_routes[i]}, '../routes/ROUTE_'+str(i)+'.kml')

        route_filename = "ROUTE_"+str(i)
        route_path = "../routes/"
        kml2geojson.main.convert(route_path+route_filename+".kml", route_path)

        with open(route_path+route_filename+".geojson", "r") as f:
            region_geojson = f.readlines()[0]
            region = json.loads(region_geojson)

        join_areas = route_geojsons[0]

        for i in route_geojsons:
            join_areas['features'].extend(i['features'])

        region['features'].extend(join_areas['features'])

        with open(route_path+route_filename+".geojson", 'w', encoding='utf8') as f:
            json.dump(region, f)

        convertGeoJson2KML(route_path+route_filename+".geojson", route_path+route_filename+".kml")
        os.remove(route_path+route_filename+".geojson")
        
    pickle.dump([Y, dict_routes, VS], open("../data/drt_data.pkl", "wb"))

    return np.array(new_VS)

def routesTransformation(dataset_path):
    df_drt = pd.read_csv(dataset_path, sep='\t')
    df_drt['ACCEPTED_TIMESTAMP'] = df_drt.ACCEPTED_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x) if not np.isnan(x) else x)
    df_drt['CANCELLED_TIMESTAMP'] = df_drt.CANCELLED_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x) if not np.isnan(x) else x)
    df_drt['DROPOFF_TIMESTAMP'] = df_drt.DROPOFF_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x))
    df_drt['PICKUP_TIMESTAMP'] = df_drt.PICKUP_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x))
    df_drt['PROPOSAL_TIMESTAMP'] = df_drt.PROPOSAL_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x))
    df_drt['REQUEST_TIMESTAMP'] = df_drt.REQUEST_TIMESTAMP.apply(lambda x: datetime.fromtimestamp(x))
    
#     df_drt = df_drt[(df_drt.STATUS == 'Finished')]
    
    df_pickup = df_drt[['RIDER_ID', 'STATUS', 'PICKUP_TIMESTAMP', 'VAN_ID', 'ORIGIN_LNG', 'ORIGIN_LAT']].rename(columns={'PICKUP_TIMESTAMP':'TIMESTAMP',
                                                                                        'ORIGIN_LNG':'LONGITUDE',
                                                                                        'ORIGIN_LAT':'LATITUDE'})
    df_pickup['TYPE'] = 'P'
    
    df_dropoff = df_drt[['RIDER_ID', 'STATUS', 'DROPOFF_TIMESTAMP', 'VAN_ID', 'DESTINATION_LNG', 'DESTINATION_LAT']].rename(columns={'DROPOFF_TIMESTAMP':'TIMESTAMP',
                                                                                                    'DESTINATION_LNG':'LONGITUDE',
                                                                                                    'DESTINATION_LAT':'LATITUDE'})
    df_dropoff['TYPE'] = 'D'
    
    df_routes = pd.concat([df_pickup,df_dropoff],ignore_index=True)
    
    df_routes['HOUR'] = df_routes.TIMESTAMP.dt.hour.astype(str)
    df_routes['DAY'] = df_routes.TIMESTAMP.dt.day.astype(str)
    df_routes['MONTH'] = df_routes.TIMESTAMP.dt.month.astype(str)
    df_routes['YEAR'] = df_routes.TIMESTAMP.dt.year.astype(str)
    df_routes['VAN_ID'] = df_routes.VAN_ID.apply(lambda x: round(float(x))).astype(str)
    
    df_routes['VAN_DAY_MONTH_YEAR'] = df_routes['VAN_ID']+ "_" + df_routes['DAY']+ "_" + df_routes['MONTH']+ "_" + df_routes['YEAR']

    df_others = df_routes[(df_routes.STATUS != 'Finished')]
    df_drt = df_routes[(df_routes.STATUS == 'Finished')]
    
    return df_drt.sort_values(by="TIMESTAMP"), df_others.sort_values(by="TIMESTAMP")


# def loadData(path, distance_grid, selected_vs_distance ,flag_process):
#     if flag_process:
#         df_proposals_shrt, requests_ids = prepProposals("../proposals/")
#         df_requests_shrt = prepRequests("../requests/", requests_ids)
#         df_rides_shrt = prepRides("../rides/")
#         df_drt_dataset = mergePropRequRides(df_proposals_shrt, df_requests_shrt, df_rides_shrt)

#         dict_routes = routesTransformation("../data/drt_dataset.csv")
        
#         areas_geojson = createGeojsonPolygon("SERVICE_AREA", distance_grid, "AREA", "../maps/")
#         getRoadsFromCenterPoints("../maps/", "CENTER", "VIRTUAL_STOPS")
        
#         dict_routes_sample = dict(random.sample(dict_routes.items(), k=25))
# #         construct_kml_list(dict_routes_sample, '../maps/ROUTES_GADRT.kml')
        
#         virtual_stops = loadVirtualStops("../maps/VIRTUAL_STOPS.geojson")
#         routes_data, dict_routes = loadRidesData('../data/routes_drt.pkl')
#         selected_virtual_stops = createRoutesWithVS("../maps/", selected_vs_distance, routes_data, dict_routes, virtual_stops)
        
#         return pickle.load(open(path, 'rb'))
#     else:
#         return pickle.load(open(path, 'rb'))
    
def process_van_routes(df_requests, df_rides, d):
    frames = []
    for van_id in tqdm(set(df_rides.VAN_ID)):
        df_van = df_rides[df_rides.VAN_ID == van_id]
        df_van = df_van.sort_values("TIMESTAMP")
        df_route = df_van.reset_index()
        new_coordinates = []
        for idx, route_stop in df_route.iterrows():
            coordinate = route_stop[['LATITUDE','LONGITUDE']].tolist()

            aux = geodesic(kilometers=d/2).destination(coordinate, -180)
            left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=d/2).destination(coordinate, 0)
            right = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(coordinate, 45)
            c_u_left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(coordinate, 135)
            c_u_right = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(coordinate, 225)
            c_d_left = [aux.latitude, aux.longitude]
            aux = geodesic(kilometers=(d/2)-0.1).destination(coordinate, 315)
            c_d_right = [aux.latitude, aux.longitude]

            coordinate_polygon = [c_u_left, c_u_right, c_d_left, c_d_right, c_u_left]
            polygon_path = mplPath.Path(coordinate_polygon)

            count_in = 0
            nearest_request = [np.nan, np.nan]
            req_d_min = 999999
            for index, request_coordinate in df_requests.iterrows():
                req_point = request_coordinate.tolist()
                if polygon_path.contains_point(req_point):
                    count_in+=1
                    req_found_d = getGeoDistanceETA_OSRM(coordinate, req_point, 5001, 'walking')[0]
                    if req_found_d < req_d_min:
                        req_d_min = req_found_d
                        nearest_request = req_point

            nearest_request.append(count_in)
            new_coordinates.append(nearest_request)

        df_new_route = pd.DataFrame(new_coordinates, columns=['LATITUDE','LONGITUDE', 'SCORE'])
        df_route[['NEW_LATITUDE', 'NEW_LONGITUDE', "SCORE"]] = df_new_route
        frames.append(df_route)
        
    pd.concat(frames).to_pickle("df_new_rides.pkl")

    return pd.concat(frames)

def load_dataset():
    df_proposals_shrt, requests_ids = prepProposals("../proposals/")
    df_requests_shrt = prepRequests("../requests/", requests_ids)
    df_rides_shrt = prepRides("../rides/")
    df_drt_dataset = mergePropRequRides(df_proposals_shrt, df_requests_shrt, df_rides_shrt)
    df_rides, df_others = routesTransformation("../data/drt_dataset.csv")
    return df_rides, df_others


def generalize_clients_requests(df_rides, min_samples):
    df_points = df_rides[['LATITUDE', 'LONGITUDE']]
    from sklearn.cluster import OPTICS
    import numpy as np
    coords = df_points.values
    X = coords
    clustering = OPTICS(min_samples=20).fit(X)
    num_clusters = len(set(clustering.labels_))
    clusters = pd.Series([coords[clustering.labels_ == n] for n in range(num_clusters)])
    print('Number of clusters: {}'.format(num_clusters))
    
    centermost_points = []
    
    for i in range(len(clusters)-1):
        centroid = (MultiPoint(clusters[i]).centroid.x, MultiPoint(clusters[i]).centroid.y)
        centermost_point = min(clusters[i], key=lambda point: great_circle(point, centroid).m)
        centermost_points.append(np.array(centermost_point))
    
    reduced_coords = np.array(centermost_points)
    df_reduced_points = pd.DataFrame(reduced_coords, columns=["LATITUDE", 'LONGITUDE'])
    
    df_reduced_points.to_pickle('generalized_requests.pkl')
    
    return df_reduced_points

def loadPrep(number_vans, init):
    df_rides, df_others = load_dataset()
    penalty_const = getPenaltyConst(5)
    df_requests_g = pd.read_pickle('generalized_requests.pkl')
    df_routes_g = pd.read_pickle('df_new_rides.pkl')
    Y = []
    for idx_top in range(init, number_vans):
        top_van = df_routes_g.groupby('VAN_ID').sum().sort_values(by='SCORE', ascending=False).reset_index().loc[idx_top]['VAN_ID']
        df_top_route = df_routes_g[df_routes_g.VAN_ID == top_van].sort_values(by='TIMESTAMP').reset_index(drop=True).drop(columns='index')
        for idx, coordinate in df_top_route.iterrows():
            if not np.isnan(coordinate['NEW_LATITUDE']):
                Y.append(np.array(coordinate[['NEW_LATITUDE', 'NEW_LONGITUDE']]))
        
    Y = np.array(Y)
    return Y