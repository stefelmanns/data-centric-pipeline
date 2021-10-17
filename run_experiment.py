import os
import yaml
import rpy2
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from helpers import read_data
from partition import split_data
from inference import find_relations


def run_experiment(setting, in_path, out_path):

	set_seed = rpy2.robjects.r('set.seed')

	with open(setting + ".yaml", 'r') as stream:
		s = argparse.Namespace(**yaml.safe_load(stream))

	setting = setting.split("/")[-1]

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

		pbar = tqdm(range(len(s.lambda_range) * len(s.components_range)),
							  desc=split)

		data_dict, header = read_data(in_path  + split + "/model/")
		part_dict, part_time = split_data(data_dict, s.non_merge_indices,
																			s.clustering_name, s.epsilon,
																			s.min_nr_samples, s.nr_clusters)

		for lambda_value in s.lambda_range:
			for nr_components in s.components_range:

				set_seed(5) # 5 12 13 14 17
				np.random.seed(10) # 10 37 22 7 40

				label = str(nr_components) + " components, lambda " +\
								'{0:.2f}'.format(lambda_value)
				name = str(nr_components) + "_" + str(int(lambda_value * 100)) +\
							 setting

				result, meta, trans_dict = find_relations(part_dict, header, nr_components, lambda_value, label, part_time)
			
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
	
	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--setting', 	type=str, default="settings/em20")
	parser.add_argument('-i', '--in_path', 	type=str, default="../lun_data_set/")
	parser.add_argument('-o', '--out_path', type=str, default="../output/")

	run_experiment(**vars(parser.parse_args()))
