import numpy as np
import pandas as pd
import sys

datafile = sys.argv[1]
dataset = pd.read_excel(datafile)
dataset.set_index("Obs", inplace=True)

Q = []
P = []
M = 0
N = 0

for i in range(len(dataset)):
	q = []
	p = []
	for j in range(0,dataset.shape[1],2):
		q.append(dataset.iloc[i,j])
		p.append(dataset.iloc[i,j+1])
	N = len(q)
	Q.append(q)
	P.append(p)
M = len(Q)

TR = np.zeros([M,M])
for i in range(M):
	for j in range(M):
		tr = 0
		for k in range(N):
			tr = tr + P[i][k] * Q[j][k]
		TR[i,j] = tr

preferences = []
for i in range(M):
	preference = []
	affordable = []
	for j in range(M):
		if (i != j):
			if (TR[i,j] <= TR[i,i]):
				preference.append(j)
	preferences.append(preference)

print("")
print("WARP")
print("")
bundles = []
for i in range(M):
	consistent = True
	for j in range(len(preferences[i])):
		violation = False
		for k in range(len(preferences[preferences[i][j]])):
			if (i == preferences[preferences[i][j]][k]):
				violation = True
				break
		if (violation):
			consistent = False
			break
	if (consistent):
		bundles.append(i)
for i in range(len(bundles)):
	change1 = False
	for j in range(len(bundles)):
		if (i != j):
			change2 = False
			for k in range(len(preferences[bundles[i]])):
				if (bundles[j] == preferences[bundles[i]][k]):
					if (i > j):
						bundle1 = bundles[i]
						bundle2 = bundles[j]
						bundles[i] = bundle2
						bundles[j] = bundle1
						change2 = True
						break
			if (change2):
				change1 = True
				break
	if (change1):
		i = -1
if (len(bundles) > 0):
	print("Bundle ordering:")
	print(bundles)
	print("")
	print("Utility maximizing bundle:")
	print(dataset.iloc[bundles[0]])
else:
	print("Can't create bundle ordering")



'''utilities = np.zeros([len(bundles)])
for i in range(len(bundles)):
	score = 0
	for j in range(len(bundles)):
		if (i != j):
			for k in range(len(preferences[bundles[i]])):
				if (bundles[j] == preferences[bundles[i]][k]):
					score = score + 1
					break
	utilities[i] = score
print("Consumption bundles:")
print(bundles)
print("")
print("Utility function:")
print(utilities)
print("")
print("Maximum utility bundle:")
maxindex = -1
maxutility = -1
for i in range(len(bundles)):
	if (utilities[i] > maxutility):
		maxindex = bundles[i]
		maxutility = utilities[i]
print(dataset.iloc[maxindex])'''

