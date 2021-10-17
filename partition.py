import time
from helpers import merge_contexts,\
										select_cluster_algorithm
from methods.cluster import cluster_from_dict


def split_data(data_dict, non_merge_indices, clustering_name, epsilon,
							 min_nr_samples, nr_clusters):

	start = time.time()

	merge_dict = merge_contexts(data_dict, non_merge_indices)

	if clustering_name != None:
		cluster_algorithm = select_cluster_algorithm(clustering_name, epsilon,
																							 	 min_nr_samples, nr_clusters)
		part_dict = cluster_from_dict(merge_dict, cluster_algorithm)

	return part_dict, time.time() - start