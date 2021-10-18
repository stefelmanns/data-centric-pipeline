import time
from helpers import merge_contexts,\
										select_cluster_algorithm
from methods.cluster import cluster_from_dict


def split_data(data_dict, non_merge_indices, cluster_name, cluster_params):

	start = time.time()

	part_dict = merge_contexts(data_dict, non_merge_indices)

	if cluster_name != None:
		cluster_algorithm = select_cluster_algorithm(cluster_name, cluster_params)
		part_dict = cluster_from_dict(part_dict, cluster_algorithm)

	return part_dict, time.time() - start