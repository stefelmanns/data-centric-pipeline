import os
import yaml
import rpy2
import pickle
import argparse

import numpy as np
from tqdm import tqdm
from main import one_run
from helpers import read_data


def run_experiment(setting, in_path, out_path):

	set_seed = rpy2.robjects.r('set.seed')

	with open(setting + ".yaml", 'r') as stream:
		experiment_dict = yaml.safe_load(stream)

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

		pbar = tqdm(experiment_dict, desc=split)

		data_dict, header = read_data(in_path + split + "/model/")

		for name, variables in experiment_dict.items():
			
			if not os.path.exists(result_path + name): os.makedirs(result_path + name)
			if not os.path.exists(meta_path + name): 	 os.makedirs(meta_path + name)
			if not os.path.exists(trans_path + name):  os.makedirs(trans_path + name)
 
			set_seed(5) # 5 12 13 14 17
			np.random.seed(10) # 10 37 22 7 40

			result, meta, trans_dict = one_run(data_dict, header, **variables)

			with open(result_path + name + "/" + split + ".pkl", 'wb') as f:
				pickle.dump(result, f)
			with open(trans_path + name + "/" + split + ".pkl", 'wb') as f:
				pickle.dump(trans_dict, f)
			with open(meta_path + name + "/" + split + ".txt", 'w') as f:
				f.write(meta)

			pbar.update(1)


if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()

	parser.add_argument('-s', '--setting', 	type=str, default="test")
	parser.add_argument('-i', '--in_path', 	type=str, default="../lun_data_set/")
	parser.add_argument('-o', '--out_path', type=str, default="../output/")

	run_experiment(**vars(parser.parse_args()))
