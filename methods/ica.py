from tqdm import tqdm
import numpy as np
import sklearn.decomposition


def warn(*args, **kwargs): pass
import warnings
warnings.warn = warn


def run_ica(data_dict, nr_components):

	ica = sklearn.decomposition.FastICA(n_components=nr_components, max_iter=2000)

	ica_dict = dict()

	# pbar = tqdm(data_dict, desc="ica")

	counter = 0
	for name, data in data_dict.items():
		if len(data.shape) == 1 or len(data) < nr_components:
		# if len(data.shape) == 1 or np.isnan(np.sum(data)):
		# if len(data.shape) == 1:
			# pbar.update(1)
			continue 
		ica_output = ica.fit_transform(data.T).T
		# print(ica.mixing_.shape)
		ica_dict[name] = ica_output
		# pbar.update(1)
		counter += 1

	return ica_dict