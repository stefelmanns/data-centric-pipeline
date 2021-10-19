import os
import pickle
import argparse
import numpy as np
from itertools import product
from sklearn.cluster import DBSCAN
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler


class RandomClusters():

	def __init__(self, nr_clusters): self.nr_clusters = nr_clusters

	def fit_predict(self, data):
		return np.random.randint(0, self.nr_clusters, len(data))


def make_paths(out_path, setting):

	out_path 	 += setting  + "/"
	result_path = out_path + "/result/"
	meta_path 	= out_path + "/meta/"
	trans_path 	= out_path + "/transformed/"

	if not os.path.exists(out_path): 		os.makedirs(out_path)
	if not os.path.exists(result_path): os.makedirs(result_path)
	if not os.path.exists(meta_path):		os.makedirs(meta_path)
	if not os.path.exists(trans_path): 	os.makedirs(trans_path)

	return argparse.Namespace(**{'result' : result_path,
															 'meta' : meta_path,
															 'trans' : trans_path})


def write_output(paths, name, split, result, meta, trans_dict):

	result_path = paths.result + name + "/"
	meta_path 	= paths.meta 	 + name + "/"
	trans_path 	= paths.trans  + name + "/"

	if not os.path.exists(result_path): os.makedirs(result_path)
	if not os.path.exists(meta_path): 	os.makedirs(meta_path)
	if not os.path.exists(trans_path):	os.makedirs(trans_path) 

	with open(result_path + split + ".pkl", 'wb') as f:	pickle.dump(result, f)
	with open(meta_path 	+ split + ".txt", 'w') as f:	f.write(meta)
	with open(trans_path 	+ split + ".pkl", 'wb') as f:	pickle.dump(trans_dict, f)


def generate_name_label(lambda_value, nr_components, cluster_params):

	name 	= "_".join([str(int(lambda_value * 100)),
										str(nr_components),
										str(cluster_params)])

	label = "\n".join(["lambda " + '{0:.2f}'.format(lambda_value),
											"components " + str(nr_components),
											"cluster " + str(cluster_params)])

	return name, label


def load_data(input_path):

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


def select_cluster_algorithm(clustering_name, cluster_params):

	if clustering_name == "dbscan":
		eps, min_nr_samples = cluster_params
		return DBSCAN(eps=eps, min_samples=min_nr_samples)
	elif clustering_name == "em":
		nr_clusters = cluster_params
		return GaussianMixture(n_components=nr_clusters)
	elif clustering_name == "random":
		nr_clusters = cluster_params
		return RandomClusters(nr_clusters)
	else:
		print("invalid clustering algorithm")


def concatenate_dict(data_dict, normalise=True):

	data_array = np.concatenate(list(data_dict.values()))

	if normalise: StandardScaler().fit_transform(data_array)

	return data_array