setting = "min_rep"

out_file = open(setting + ".yaml", 'w')

for i in range(10, 35):
# for i in ["null"]:
	for j in range(1, 11):

		name = str(i) + "_" + str(j) + "_" + setting
		# name = str(j) + "_conventional"
		label = "\"" + str(i) +  " components, lambda " + '{0:.2f}'.format(j*0.01) + "\""
		# label = "\"0 components, lambda " + '{0:.2f}'.format(j) + "\""
		clustering_name = "null"
		epsilon = "null"
		min_nr_samples = "null"
		nr_clusters = "null"
		nr_components = i
		lambda_value = j * 0.01
		non_merge_indices = {1, 2}

		output_string = name + " : " +\
										"\n  " + "label : " + label +\
										"\n  " + "clustering_name : " + str(clustering_name) +\
										"\n  " + "epsilon : " + str(epsilon) +\
										"\n  " + "min_nr_samples : " + str(min_nr_samples) +\
										"\n  " + "nr_clusters : " + str(nr_clusters) +\
										"\n  " + "nr_components : " + str(nr_components) +\
										"\n  " + "lambda_value : " + str(lambda_value) +\
										"\n  " + "non_merge_indices : " + str(non_merge_indices) +\
										"\n\n"

		print(output_string)
		out_file.write(output_string)

out_file.close()