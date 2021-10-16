import time
from helpers import merge_contexts,\
										select_cluster_algorithm,\
										concatenate_dict
from methods.ica import run_ica
from methods.glasso import run_glasso
from methods.cluster import cluster_from_dict


def one_run(data_dict, header, non_merge_indices, clustering_name, epsilon,
						min_nr_samples, nr_clusters, nr_components, lambda_value, label):

	start = time.time()

	data_dict = merge_contexts(data_dict, non_merge_indices)

	if clustering_name != None:
		cluster_algorithm = select_cluster_algorithm(clustering_name, epsilon,
																							 	 min_nr_samples, nr_clusters)
		data_dict = cluster_from_dict(data_dict, cluster_algorithm)

	if nr_components != None: data_dict = run_ica(data_dict, nr_components)

	data_array = concatenate_dict(data_dict)

	nr_data_points = len(data_array)

	result = run_glasso(data_array, lambda_value, header)

	metadata = label + "\nnr datapoints " + str(nr_data_points) + "\nduration " +\
						 str(time.time() - start)

	return result, metadata, data_dict