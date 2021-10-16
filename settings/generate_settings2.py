
out_file = open("Settings/em.yaml", 'w')

counter = 0

for i in range(20, 31):
	for j in range(1, 9):
		for k in range(20, 21):
		# for k in range(20, 81, 20):

			name = str(i) + "_" + str(j) + "_" + str(k) + "_em"
			label = "\"" + str(i) +  " components, lambda " +\
						  '{0:.2f}'.format(0.01 * j) + ", clusters " + str(k) + "\""
			clustering_name = "em"
			epsilon = "null"
			min_nr_samples = "null"
			nr_clusters = k
			nr_components = i
			lambda_value = 0.01 * j
			non_merge_indices = {}

			out_string = name + " : " +\
									 "\n  " + "label : " + label +\
									 "\n  " + "clustering_name : " + str(clustering_name) +\
									 "\n  " + "epsilon : " + str(epsilon) +\
									 "\n  " + "min_nr_samples : " + str(min_nr_samples) +\
									 "\n  " + "nr_clusters : " + str(nr_clusters) +\
									 "\n  " + "nr_components : " + str(nr_components) +\
									 "\n  " + "lambda_value : " + str(lambda_value) +\
									 "\n  " + "non_merge_indices : " + str(non_merge_indices) +\
									 "\n\n"

			print(out_string)
			out_file.write(out_string)
			counter += 1

out_file.close()
print(counter) 