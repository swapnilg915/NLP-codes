#### A. Levenshtein Distance 

def levenshtein(s1,s2):
	if len(s1) > len(s2):
		s1,s2= s2,s1

	dist = range(len(s1) + 1)
	for index2, char2 in enumerate(s2):
		newDistances = [index2 + 1]
		for index1,char1 in enumerate(s1):
			if char1 == char2:
				newDistances.append(dist[index1])
			else:
				newDistances.append(1 + min((dist[index1], dist[index1 + 1], newDistances[-1])))

		dist = newDistances
	return dist[-1]

print(levenshtein("what is javascript","what is java"))


