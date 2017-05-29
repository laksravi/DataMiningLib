import sys
import csv
import numpy as np

# This file acts as a decision tree classifier

testing_file_name = sys.argv[1]
testing_file_handler = open(testing_file_name, mode='r')
testing_file_csv = csv.DictReader(testing_file_handler, )
for line in testing_file_csv:
	features =[line['Attr1'], line['Attr2'], line['Attr3'], line['Attr3']]
	features = np.array(features).astype(dtype=float)
	if features[1] <7.94:
		if features[1] <5.0:
			if features[2] <5.95:
				if features[1] <4.91:
					print("1")
				else:
					print("0")
			else:
				if features[2] <6.385:
					if features[1] <3.01:
						print("1")
					else:
						print("0")
				else:
					print("0")
		else:
			print("1")
	else:
		print("0")
