import math
import json
from collections import Counter

import pyproj
import pandas as pd
from functools import partial
from shapely.geometry import Point, LineString
from shapely.ops import transform


'''
def: genrate gps trace data from vehicles
note: any vehicle whose entries are seperated by long time delay is a new path
usage: filepath - string path of data in csv format
usage: min_interval - minimum time interval for path seperation
'''
def get_vehicles_data(filepath: str, min_interval=60):

	gps_trace = pd.read_csv(filepath)
	gps_trace.sort_values(['VehicleID','TimeStamp'], inplace=True)
	gps_trace['Timestamp_diff'] = gps_trace.groupby('VehicleID')['TimeStamp'].diff()
	gps_trace = gps_trace.fillna({"Timestamp_diff":0})

	## adding Group Id to signify path
	prev_vid = None
	prev_gid = 0
	gids = []
	for _, row in gps_trace.iterrows():
	    
	    if row['VehicleID'] != prev_vid:
	        prev_gid += 1
	    if row['Timestamp_diff'] > min_interval:
	        prev_gid += 1
	    
	    prev_vid = row['VehicleID']
	    gids.append(prev_gid)

	gps_trace['Gid'] = gids
	return gps_trace

'''
def: creates mapping of road id , coordinates and road direction for reference
usage: filepath - string path of data in json format
'''
def get_roads_data(filepath: str):

	with open(filepath, 'r') as f:
		road_geometry = json.load(f)

	roads_data = {}
	for props in road_geometry['features']:
		roads_data[props['id']] = {}
		roads_data[props['id']]['coordinates'] = props['geometry']['coordinates']

		roads_data[props['id']]['direction'] = props['properties']['direction']

	return roads_data

'''
def: convert lat-long to  utm coordinates
usage: df - dataframe to convert
usage: lat_col - latitude column
usage: long_col - longitude column
'''
def transform_to_utm(df: pd.DataFrame, lat_col='Lattitude', long_col='Longitude'):

	df['geometry'] = df.apply(lambda row: Point(row[long_col], row[lat_col]), axis=1)
	project = partial(
	pyproj.transform,
	pyproj.Proj(init='epsg:4326'), # source coordinate system
	pyproj.Proj('+proj=utm +zone=10 +ellps=WGS84 +datum=WGS84 +units=m +no_defs')) # destination coordinate system
	converted = df.copy()
	converted['utm'] = converted.geometry.apply(lambda row: transform(project,row))
	return converted

'''
def: clean trace, remove broken data
usage: transformed_df - dataframe
'''
def clean_gps_trace(transformed_df: pd.DataFrame):
	all_paths = list(pd.unique(transformed_df['Gid'].values))
	updated_df = pd.DataFrame()

	for gid in all_paths:
		sub = transformed_df[transformed_df['Gid'] == gid]
		if len(sub) == 1:
			continue
		if all(sub['Speed'] == 0):
			continue

		updated_df = updated_df.append(sub)
	if len(updated_df) > 0:
		
		valid_paths = list(pd.unique(updated_df['Gid'].values))
		return updated_df, valid_paths
	else:
		return None, None

'''
def: generate coordinate mapping per path
usage: df - dataframe
'''
def generate_path_coordinates(df: pd.DataFrame):
	path_coordinate_mapping = {}
	all_paths = list(pd.unique(df['Gid'].values))
	for gid in all_paths:
		subset = df[df['Gid'] == gid]
		points = []
		for val in subset['utm'].tolist():
			points.append([val.x, val.y])
		path_coordinate_mapping[gid] = points
	return path_coordinate_mapping

'''
def: compute hausdorff distance between two linestrings
usage: list_of_points_1 - list of points represnting linestring
usage: list_of_points_2 - list of points represnting linestring
'''
def compute_distance(list_of_points_1: list, list_of_points_2: list):
	p1 = LineString(list_of_points_1)
	p2 = LineString(list_of_points_2)
	return p1.hausdorff_distance(p2)

'''
def: generates final output report with lanes for each road id. this writes a csv
usage: roads_data - given roads data
usage: master_dict - extracted lanes count mapping
'''
def generate_report(roads_data: dict, master_dict: dict):
	roads = []
	road_direction = []
	total_lanes = []
	backward_lanes = []
	forward_lanes = []
	
	for road, features in roads_data.items():
		direction = features['direction']
		if direction == 'Both':
			num_forward = master_dict[int(road)]
			num_backward = master_dict[-int(road)]
			num_total = num_forward + num_backward
		else:
			if int(road) not in master_dict:
				key_to_check = - int(road)
			else:
				key_to_check = int(road)
			
			if direction == 'Forward':
				num_forward = master_dict[key_to_check]
				num_backward = 0
				num_total = num_forward
			else:
				num_backward = master_dict[key_to_check]
				num_forward = 0
				num_total = num_backward
			
		roads.append(road)
		road_direction.append(direction)
		total_lanes.append(num_total)
		backward_lanes.append(num_backward)
		forward_lanes.append(num_forward)
	outcome = pd.DataFrame({'RoadID': roads,
							'DirectionOfRoad': road_direction,
							'Number_of_lanes': total_lanes,
							'Number_of_Backward_lanes': backward_lanes,
							'Number_of_Forward_lanes': forward_lanes})

	return outcome