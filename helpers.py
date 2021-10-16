import os
import numpy as np
from tqdm import tqdm
# TODO import specific function
import sklearn.cluster
import sklearn.mixture
import sklearn.preprocessing
from itertools import product


class RandomClusters():

	def __init__(self, nr_clusters): self.nr_clusters = nr_clusters

	def fit_predict(self, data):		 return np.random.randint(0, self.nr_clusters, len(data))


def read_data(input_path):

	data_dict = dict()

	for file_name in os.listdir(input_path):

		if file_name == ".DS_Store": continue
		file_path = input_path + file_name

		with open(file_path, 'r') as f:
			header = f.readline().strip("\n").split(",")
			GFP_index = header.index("GFP")
			header.remove("GFP")

		data_array = np.loadtxt(file_path, delimiter=",", skiprows=1)
		if len(data_array.shape) == 1: continue 

		data_array = np.delete(data_array, GFP_index, axis=1)

		data_dict[file_name.strip(".csv")] = data_array

	return data_dict, header


def merge_contexts(in_data_dict, non_merge_indices):

	non_merge_list = list()	
	for index in non_merge_indices:
		non_merge_list.append({key.split("_")[index] for key in in_data_dict})
	
	non_merge_keys = set(product(*non_merge_list))

	out_data_dict = dict()

	for keys in non_merge_keys:

		array_list = list()

		for key2, data_array in in_data_dict.items():
			if all(key1 in key2.split("_") for key1 in keys):
				array_list.append(data_array)

		if len(array_list) > 0: out_data_dict["_".join(keys)] = np.concatenate(array_list)

	return out_data_dict


def select_cluster_algorithm(clustering_name, epsilon,	min_nr_samples, nr_clusters):

	if clustering_name == "dbscan":
		return sklearn.cluster.DBSCAN(eps=epsilon, min_samples=min_nr_samples)
	elif clustering_name == "em":
		return sklearn.mixture.GaussianMixture(n_components=nr_clusters)
	elif clustering_name == "random":
		return RandomClusters(nr_clusters)
	else:
		print("invalid clustering algorithm")


def concatenate_dict(data_dict, normalise=True):

	data_array = np.concatenate(list(data_dict.values()))

	if normalise: sklearn.preprocessing.StandardScaler().fit_transform(data_array)

	return data_array


def create_name(non_merge_indices, clustering_name, epsilon, min_nr_samples, nr_clusters, nr_components, lambda_value):

	name = str(non_merge_indices)

	name += clustering_name

	if clustering_name == "dbscan":
		name += "_" + str(epsilon)
		name += "_" + str(min_nr_samples)
	else:
		name += "_" + str(nr_clusters)

	name += "_ica" + str(nr_components) 

	name += "_glasso" + str(lambda_value)

	return name