import numpy as np
from tqdm import tqdm


def cluster_from_array(data_array, cluster_algorithm):

	if len(data_array.shape) == 1: return np.array([])

	cluster_dict = dict()

	labels = cluster_algorithm.fit_predict(data_array)

	per_cluster = {label : list() for label in set(labels)}

	for i, label in enumerate(labels):
		per_cluster[label].append(np.reshape(np.array(data_array[i]), (1, len(data_array[i]))))

	for label in per_cluster:
		if len(per_cluster[label]) > 1 and label != -1:
			cluster_dict[str(label)] = np.concatenate(per_cluster[label], axis=0)

	return cluster_dict


def cluster_from_dict(data_dict, cluster_algorithm):

	# pbar = tqdm(data_dict.values(), desc="clustering")

	cluster_dict = dict()

	for key, data_array in data_dict.items():

		sub_dict = cluster_from_array(data_array, cluster_algorithm)

		for label, cluster in sub_dict.items():
			cluster_dict[key + "_" + label] = cluster

		# pbar.update(1)

	return cluster_dict

