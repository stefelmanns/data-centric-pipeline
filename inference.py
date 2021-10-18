import time
from helpers import concatenate_dict
from methods.ica import run_ica
from methods.glasso import run_glasso


def find_relations(part_dict, header, nr_components, lambda_value, label, 
									 part_time):

	start = time.time()

	if nr_components != None: part_dict = run_ica(part_dict, nr_components)

	data_array = concatenate_dict(part_dict)

	nr_data_points = len(data_array)

	result = run_glasso(data_array, lambda_value, header)

	meta = label + "\nnr datapoints " + str(nr_data_points) + "\nduration " +\
				 str(part_time + time.time() - start)

	return result, meta, part_dict