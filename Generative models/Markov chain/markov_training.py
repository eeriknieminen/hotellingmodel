import numpy as np
import random
import sys
from threading import Thread

def finduniquewords(line, uniquewords):
	for i in range(len(line)):
		word = line[i]
		unique = 1
		for j in range(len(uniquewords)):
			match = uniquewords[j]
			if (word == match):
				unique = 0
				break
		if (unique == 1):
			uniquewords.append(word)

def calculatenextwordfrequencies(index, data, uniquewords, nextword_frequencies):
	word = uniquewords[index]
	for i in range(len(data)):
		for j in range(len(data[i])):
			match = data[i][j]
			if (word == match and j < len(data[i])-1):
				nextword = data[i][j+1]
				nextword_index = uniquewords.index(nextword)
				nextword_frequencies[index,nextword_index] = nextword_frequencies[index,nextword_index] + 1000

def calculatenextwordprobabilities(index, uniquewords_len, nextword_frequencies, nextword_probabilities):
	total = sum(nextword_frequencies[index])
	for i in range(uniquewords_len):
		nextword_probabilities[index][i] = nextword_frequencies[index][i] / total

if (len(sys.argv) > 3):

	data = []
	l = 0
	maxl = int(sys.argv[2])
	with open(sys.argv[1], 'r', encoding='UTF-8') as file:
		for line in file:
			data.append(line.replace("\n","").split(" "))
			l = l+1
			if (l == maxl):
				break

	uniquewords = []
	threads = []
	for i in range(len(data)):
		t = Thread(target=finduniquewords, args=(data[i],uniquewords,))
		threads.append(t)
		t.start()
	for i in range(len(data)):
		threads[i].join()
	#print(uniquewords)

	nextword_frequencies = np.ones(shape=[len(uniquewords),len(uniquewords)])
	threads = []
	for i in range(len(uniquewords)):
		t = Thread(target=calculatenextwordfrequencies, args=(i,data,uniquewords,nextword_frequencies))
		threads.append(t)
		t.start()
	for i in range(len(uniquewords)):
		threads[i].join()
	#print(nextword_frequencies)

	nextword_probabilities = np.zeros(shape=[len(uniquewords),len(uniquewords)])
	threads = []
	for i in range(len(uniquewords)):
		t = Thread(target=calculatenextwordprobabilities, args=(i,len(uniquewords),nextword_frequencies, nextword_probabilities))
		threads.append(t)
		t.start()
	for i in range(len(uniquewords)):
		threads[i].join()
	#print(nextword_probabilities)

	with open(sys.argv[3], "w") as file:
		for i in range(len(uniquewords)):
			string = ","
			for j in range(len(uniquewords)):
				string = string + "," + str(round(nextword_probabilities[i][j]*100000)/100000)
			file.write(uniquewords[i] + "	" + string[2:] + "\n")