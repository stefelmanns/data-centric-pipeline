import os
import yaml
import rpy2
import argparse

import numpy as np
from tqdm import tqdm
from itertools import product

from partition import split_data
from inference import find_relations
from helpers import load_data, make_paths, write_output, generate_name_label


def run_experiment(setting, in_path, out_path):

	set_seed = rpy2.robjects.r('set.seed')

	with open(setting + ".yaml", 'r') as stream:
		s = argparse.Namespace(**yaml.safe_load(stream))

	setting = setting.split("/")[-1]

	inference_params = list(product(s.components_list, s.lambda_list))

	paths = make_paths(out_path, setting)

	print("running ", setting)

	for split in os.listdir(in_path):

		if split == ".DS_Store": continue

		data_dict, header = load_data(in_path  + split + "/model/")
	
		pbar = tqdm(range(len(inference_params) * len(s.cluster_hyper_list)),
								desc=split)

		for cluster_params in s.cluster_hyper_list:

			np.random.seed(10)

			part_dict, part_time = split_data(data_dict, s.non_merge_indices,
																				s.cluster_name, cluster_params)

			for params in inference_params:

				nr_components, lambda_value = params

				name, label = generate_name_label(lambda_value, nr_components, 
																					cluster_params)

				set_seed(17), np.random.seed(10)

				result, meta, trans_dict = find_relations(part_dict, header, 
																									nr_components, lambda_value, 
																									label, part_time)
			
				write_output(paths, name, split, result, meta, trans_dict)

				pbar.update(1)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--setting', 	type=str, default="settings/em")
	parser.add_argument('-i', '--in_path', 	type=str, default="../lun_data_set/")
	parser.add_argument('-o', '--out_path', type=str, default="../output/")

	run_experiment(**vars(parser.parse_args()))
