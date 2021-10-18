import os
import yaml
import rpy2
import pickle
from argparse import ArgumentParser,\
										 Namespace

import numpy as np
from tqdm import tqdm
from itertools import product

from helpers import load_data
from partition import split_data
from inference import find_relations


def run_experiment(setting, in_path, out_path):

	set_seed = rpy2.robjects.r('set.seed')

	with open(setting + ".yaml", 'r') as stream:
		s = Namespace(**yaml.safe_load(stream))

	setting = setting.split("/")[-1]

	inference_params = list(product(s.components_list, s.lambda_list))

	out_path 	 += setting + "/"
	result_path = out_path +  "/result/"
	meta_path 	= out_path + "/meta/"
	trans_path 	= out_path + "/transformed/"
	if not os.path.exists(out_path): 		os.makedirs(out_path)
	if not os.path.exists(result_path): os.makedirs(result_path)
	if not os.path.exists(meta_path):		os.makedirs(meta_path)
	if not os.path.exists(trans_path): 	os.makedirs(trans_path)
	
	for split in os.listdir(in_path):

		if split == ".DS_Store": continue

		print("loading data...")
		data_dict, header = load_data(in_path  + split + "/model/")
	
		pbar = tqdm(range(len(inference_params) * len(s.cluster_hyper_list)),
								desc=split)

		for cluster_params in s.cluster_hyper_list:

			part_dict, part_time = split_data(data_dict, s.non_merge_indices,
																				s.cluster_name, cluster_params)

			for params in inference_params:

				nr_components, lambda_value = params

				set_seed(5) # 5 12 13 14 17
				np.random.seed(10) # 10 37 22 7 40

				
				if s.cluster_name == None:
					name = str(nr_components) + "_" + str(int(lambda_value * 100))
					label = str(nr_components) + " components, lambda " +\
									'{0:.2f}'.format(lambda_value)
				else:
					name = str(cluster_params) + "_" + str(nr_components) + "_" +\
								str(int(lambda_value * 100))
					label = "cluster parameters " + str(cluster_params) +\
									str(nr_components) + " components, lambda " +\
									'{0:.2f}'.format(lambda_value)
					

				result, meta, trans_dict = find_relations(part_dict, header, 
																									nr_components, lambda_value, 
																									label, part_time)
			
				if not os.path.exists(result_path+name): os.makedirs(result_path+name)
				if not os.path.exists(meta_path+name): 	 os.makedirs(meta_path+name)
				if not os.path.exists(trans_path+name):  os.makedirs(trans_path+name)

				with open(result_path + name + "/" + split + ".pkl", 'wb') as f:
					pickle.dump(result, f)
				with open(trans_path + name + "/" + split + ".pkl", 'wb') as f:
					pickle.dump(trans_dict, f)
				with open(meta_path + name + "/" + split + ".txt", 'w') as f:
					f.write(meta)

				pbar.update(1)


if __name__ == '__main__':
	
	parser = ArgumentParser()

	parser.add_argument('-s', '--setting', 	type=str, default="settings/em")
	parser.add_argument('-i', '--in_path', 	type=str, default="../lun_data_set/")
	parser.add_argument('-o', '--out_path', type=str, default="../output/")

	run_experiment(**vars(parser.parse_args()))
