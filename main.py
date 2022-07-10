import re
import sys
import json
import warnings

import hdbscan
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

import helper

warnings.filterwarnings("ignore")

'''
dbscan clustering which calls a naive method to tune hyperparameters.
A more ideal hyperparameter tuning involves other methods which would need more time

'''
def clustering_model(distance_matrix: np.ndarray, scale=True):

	if scale:
		distance_matrix = MinMaxScaler().fit_transform(distance_matrix)

	alpha = np.median(distance_matrix)
	clusterer = hdbscan.HDBSCAN(metric='precomputed', min_samples=1, alpha=alpha)
	output = clusterer.fit(distance_matrix)
	num_lanes = clusterer.labels_.max()
	if num_lanes <= 0: #incase of no prediction
		num_lanes = 1
	return num_lanes

'''
generates a distance matrix of linestring hausdorff distances
between each vehicle. the shape is num_vehicles x num_vehicles
'''
def generate_distance_matrix(updated_df: pd.DataFrame, valid_paths: list):

	dist_matrix = np.zeros((len(valid_paths),len(valid_paths)))
	path_ids_dict = {i:gid for i,gid in enumerate(valid_paths)}

	path_coordinate_mapping = helper.generate_path_coordinates(updated_df)
	for i in range(dist_matrix.shape[0]):
		for j in range(dist_matrix.shape[1]):
			if i == j:
				dist_matrix[i][j] = 0
			elif i<j:
				v1 = path_ids_dict[i]
				v2 = path_ids_dict[j]
				dist_matrix[i][j] = helper.compute_distance(path_coordinate_mapping[v1],path_coordinate_mapping[v2])
				dist_matrix[j][i] = dist_matrix[i][j]
			else:
				pass

	return dist_matrix

'''
function that transforms coordinates to utm, removes stationary vehicles and those with one obs and
creates clusters after generating a distance matrix using hausdorff distance 
'''
def analyse_road(vehicles_data: pd.DataFrame, road_id: int):

	on_road = vehicles_data[vehicles_data['DirOfTravel'] == road_id]
	on_road_transformed = helper.transform_to_utm(on_road)
	on_road_transformed, valid_paths = helper.clean_gps_trace(on_road_transformed)

	if valid_paths is None:
		num_lanes = 1
		return num_lanes

	dist_matrix = generate_distance_matrix(on_road_transformed, valid_paths)
	num_lanes = clustering_model(dist_matrix)
	return num_lanes


''' 
entry function that obtains each lane info and then calls the road analysis function. 
It updated a dictionary with predicted lanes for each direction of travel
'''
def process_roads(vehicles_data: pd.DataFrame):

	master_dict = {}
	unique_roads = list(pd.unique(vehicles_data['DirOfTravel'].values))
	for road_id in unique_roads:
		print("Processing " + str(road_id))
		num_lanes = analyse_road(vehicles_data, road_id)
		master_dict[road_id] = num_lanes

		
	return master_dict

'''
bring everything together
'''
def main(vehicles_filepath: str, roads_filepath: str, output_filepath: str):

	vehicles_data = helper.get_vehicles_data(vehicles_filepath)
	roads_data = helper.get_roads_data(roads_filepath)

	master_dict = process_roads(vehicles_data)
	
	print("Generating report..!")
	output = helper.generate_report(roads_data, master_dict)
	output.to_csv(output_filepath, index=False)

if __name__ == '__main__':

	if len(sys.argv)!=4:
		print('usage: python main.py <vehicles_filepath> <roads_filepath> <out_filepath>')
	
	vehicles_filepath = sys.argv[1]
	roads_filepath = sys.argv[2]
	out_filepath = sys.argv[3]
	main(vehicles_filepath, roads_filepath, out_filepath)