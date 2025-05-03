import numpy as np
import random
import sys
from threading import Thread

if (len(sys.argv) > 2):

	lines = open(sys.argv[1], 'r').readlines()
	uniquewords_len = len(lines)
	uniquewords = []
	nextword_probabilities = np.zeros(shape=[uniquewords_len,uniquewords_len])
	for i in range(uniquewords_len):
		array = lines[i].split("	")
		uniquewords.append(array[0])
		nextword_probabilities[i] = array[1].split(",")

	length = int(sys.argv[2])
	word = uniquewords[round(random.random()*(len(uniquewords)-1))]
	string = word + " "
	for i in range(length-1):
		index = uniquewords.index(word)
		x = random.random()
		cumulativeprobability = 0
		ready = 0
		for j in range(len(uniquewords)):
			cumulativeprobability = cumulativeprobability + nextword_probabilities[index][j]
			if (x <= cumulativeprobability):
				word = uniquewords[j]
				ready = 1
				break
		if (ready == 0):
			word = uniquewords[round(random.random()*(len(uniquewords)-1))]
		string = string + word + " "
	print(string)