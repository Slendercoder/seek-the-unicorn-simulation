import numpy as np

class Player:

	def __init__(self, region, score, overlapping):
		self.region = region
		self.score = score
		self.overlapping = overlapping
		self.biases = [0.6] + [0.05] * 8
		self.p_stubborness = [7, 1000, 4]
	
	def sigmoid(self, x):
		beta = self.p_stubborness[1]
		gamma = self.p_stubborness[2]
		return 1. / (1 + np.exp(-beta * (x - gamma)))
	
	def attractiveness(self, focal):
		alpha = self.p_stubborness[0]
		region_from = self.region
		stubborness = 0
		if (region_from != 0) and (region_from == focal):
			stubborness = alpha * self.sigmoid(self.score)
		return self.biases[focal] + stubborness

	def probabilities(self):
		attracts = [self.attractiveness(k) for k in range(9)]
		sum = np.sum(attracts)
		return [x/sum for x in attracts]


pl1 = Player(1, 29, [0])
probs = pl1.probabilities()
lista = []
for n in range(100):
	p = np.random.choice(range(9), p=probs)
	lista.append(p)

print(np.mean(lista))
