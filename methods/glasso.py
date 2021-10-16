# TODO clean import section
from rpy2.robjects.packages import importr
import rpy2.robjects.numpy2ri
import rpy2.robjects as robjects
rpy2.robjects.numpy2ri.activate()
huge = importr('huge')

def run_glasso(data_array, lambda_value, header):

	# TODO consider using 'huge.glasso()'
	R_output = huge.huge(data_array, lambda_value, method='glasso',
											 verbose=False)

	python_output = { key : R_output.rx2(key) for key in R_output.names }

	result = dict()
	for i1, var1 in enumerate(header):
		for i2, var2 in enumerate(header):
			if var1 != var2: result[(var1, var2)] = python_output['icov'][0][i1, i2]

	return result