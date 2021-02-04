# Class definition for solving "Seeking the unicorn" task
# Edgar Andrade-Lotero 2020
# Run with Python 3
# Run from main.py
# Requires FRA.py

from random import choice, choices, uniform, random, sample, randint
from math import floor
import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
import FRA
import os

DEB = False
IMPR = False
TO_FILE = True

CONTINUO = False
CONTADOR = 1
TOLERANCIA = 1

#################################
# FUNCTIONS
################################

# Define players
class player :
	'''Object defining a player.'''

	def __init__(self, Ready, Decision, Choice, Where, Joint, Score, Accuracy, Name, modelParameters):
		self.ready = Ready
		self.decision = Decision
		self.choice = Choice
		self.where = Where
		self.joint = Joint
		self.score = Score
		self.accuracy = Accuracy
		self.name = Name
		self.parameters = modelParameters
		self.regionsNames = ['RS', \
		           'ALL', \
		           'NOTHING', \
		           'BOTTOM', \
		           'TOP', \
		           'LEFT', \
		           'RIGHT', \
		           'IN', \
		           'OUT']
		self.regionsCoded = ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
		                  '', # NOTHING
		                  'GHIJKLMNOPQRSTUVWXYZ0123456789;:', # BOTTOM
		                  'abcdefghijklmnopqrstuvwxyzABCDEF', # TOP
		                  'abcdijklqrstyzABGHIJOPQRWXYZ4567', # LEFT
		                  'efghmnopuvwxCDEFKLMNSTUV012389;:', # RIGHT
		                  'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012', # IN
		                  'abcdefghipqxyFGNOVW3456789;:' # OUT
		                  ]
		self.strategies = [FRA.lettercode2Strategy(x, 8) for x in self.regionsCoded]
		self.regions = [FRA.code2Vector(x, 8) for x in self.strategies]
		self.complements = [[1 - x for x in sublist] for sublist in self.regions]

	def make_decision(self):
		attractiveness = self.attract()
		sum = np.sum(attractiveness)
		probabilities = [x/sum for x in attractiveness]
		# newChoice = choices(range(9), weights=probabilities)[0]
		newChoice = np.random.choice(range(9), p=probabilities)
		self.choice = newChoice

	def attract(self, DEB=False):
		wALL = float(self.parameters[0])
		wNOTHING = float(self.parameters[1])
		wBOTTOM = float(self.parameters[2])
		wTOP = float(self.parameters[2])
		wLEFT = float(self.parameters[2])
		wRIGHT = float(self.parameters[2])
		wIN = float(self.parameters[3])
		wOUT = float(self.parameters[3])
		wRS = 1 - np.sum(np.array([wALL, wNOTHING, wBOTTOM, wTOP, wLEFT, wRIGHT, wIN, wOUT]))
		assert(wRS > 0), "Incorrect biases!"

		alpha = float(self.parameters[4]) # for how much the focal region augments attractiveness
		beta = float(self.parameters[5]) # amplitude of the WSLS sigmoid function
		gamma = float(self.parameters[6]) # position of the WSLS sigmoid function

		delta = float(self.parameters[7]) # for how much the added FRA similarities augments attractiveness
		epsilon = float(self.parameters[8]) # amplitude of the FRA sigmoid function
		zeta = float(self.parameters[9]) # position of the FRA sigmoid function

		# start from biases
		attractiveness = [wRS, wALL, wNOTHING, wBOTTOM, wTOP, wLEFT, wRIGHT, wIN, wOUT]
		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('Player', pl)
			print('attractiveness before WS and FRA\n', attactPrint)

		# Adding 'Win Stay'
		if self.choice != 0:
			attractiveness[self.choice] += alpha * FRA.sigmoid(self.score, beta, gamma)

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with WS\n', attactPrint)

		# Adding 'FRA'
		visited = FRA.code2Vector(self.where, 8)
		sims1 = [0] + [FRA.sim_consist(visited, x) for x in self.regions]
		overlap = FRA.code2Vector(self.joint, 8)
		sims2 = [0] + [FRA.sim_consist(overlap, x) for x in self.complements]
		sims2[0] = 0 # ALL's complement, NOTHING, does not repel to ALL
		FRAsims = np.add(sims1, sims2)
		attractiveness = np.add(attractiveness, [delta * FRA.sigmoid(x, epsilon, zeta) for x in FRAsims])

		if DEB:
			attactPrint = ["%.3f" % v for v in attractiveness]
			print('attractiveness with FRA\n', attactPrint)

		return attractiveness

# Define Experiment Object
class Experiment :
	'''Object defining the experiment and simulation'''

	def __init__(self, gameParameters, modelParameters, shaky_hand=0):
		assert(len(gameParameters) == 5), "Game parameters incorrect length!"
		self.gameParameters = gameParameters
		self.modelParameters = modelParameters
		self.shaky_hand = shaky_hand
		# Create data frame
		cols = ['Dyad', 'Round', 'Player', 'Answer', 'Time']
		cols += ['a' + str(i+1) + str(j+1) for i in range(0, 8) for j in range(0, 8)]
		cols += ['Score', 'Joint', 'Is_there', 'where_x', 'where_y', 'Strategy']
		self.df = pd.DataFrame(columns=cols)

	def run_dyad(self, TO_FILE=True):

		p = self.gameParameters[0] # probability of there being a unicorn (usually 0.5)
		Pl = self.gameParameters[1] # number of players (usually 2)
		Num_Loc = self.gameParameters[2] # number of locations (squares in a row in the grid; usually 8)
		N = self.gameParameters[3] # number of iterations per experiment

		# Create players
		Players = []
		for k in range(0, Pl):
			Players.append(player(False, "", 0, [], [], 0, False, int(uniform(0, 1000000)), self.modelParameters))

		# Create dyad name
		dyad = str(Players[0].name)[:5] + str(Players[1].name)[:5]

		# Start the rounds
		for i in range(0, N):

			# print "----------------------------"
			# print "Now playing round " + str(i)
			# print "----------------------------"

			#Initializing players for round
			for pl in Players:
				pl.decision = ""
				pl.where = []
				pl.joint = []
				pl.ready = False
				pl.score = 0
				pl.accuracy = False

			# Initializing the board
			Board = [0 for l in range(0, Num_Loc * Num_Loc)]

			# Determine whether there is a unicorn and where
			place = -1
			if uniform(0, 1) > p:
				place = int(floor(uniform(0, Num_Loc * Num_Loc - 1)))
				Board[place] = 1

			# Determine players' chosen region
			estrat = []
			for k in range(0, Pl):
				chosen = Players[k].choice
				if chosen == 0:
				    n = randint(2, Num_Loc * Num_Loc - 2)
				    estrat.append(list(np.random.choice(Num_Loc * Num_Loc, n, replace=False)))
				else:
					estrat.append(Players[k].strategies[chosen - 1])
				estrat[k] = self.shake(estrat[k])
				# print(f"Player's {k} chosen region ({chosen}):")
				# print(estrat[k])
				# FRA.imprime_region(FRA.code2Vector(estrat[k], Num_Loc))

			# Start searching for the unicorn
			for j in range(0, Num_Loc * Num_Loc + 1):
				# print("\nRunning iteration " + str(j))
				for k in range(0, Pl):
					# See if other player said present. If so, do the same
					if Players[1 - k].decision == "Present":
						Players[k].decision = "Present"
						# print("Player " + str(k) + " said Present")
						Players[k].ready = True
						break
					# If the other player did not say Present, and
					# current player is not ready, then...
					elif not Players[k].ready:
						# ...look at the location determined by the strategy
		#				print("Player " + str(k) + " is using strategy: " + \
		#					FRA.nameRegion(Players[k].choice))
		#				print("He is looking at location: " + str(strategies[Players[k].strategy]))
						# See if the strategy is not over...
						if j<len(estrat[k]):
							search_place = estrat[k][j]
							Players[k].where.append(search_place)
							# print("Player " + str(k) + " is searching at " + str(search_place))
							if Board[search_place] == 1:
								Players[k].decision = "Present"
								# print("Player " + str(k) + " said Present")
								Players[k].ready = True
							# else: print("Player " + str(k) + " found no unicorn")
						# Otherwise, say Absent
						else:
							# The strategy is over, so bet for Absent
							Players[k].decision = "Absent"
							# print("Player " + str(k) + " said Absent")
							Players[k].ready = True
					# Chechk if both players are ready. If so, stop search
					elif Players[1-k].ready == True:
						break
				else: continue
				break

			# print("\n")

			# Determine locations visited by both players
			# both = [x for x in Players[0].where if x in Players[1].where]
			both = list(set(Players[0].where).intersection(set(Players[1].where)))
			# print("Locations checked by both players: " + str(both))
			# print("The players checked on the same locations " + str(len(both)) + " times")

			# Create row of data as dictionary
			row_of_data = {}

			# Save data per player
			for k in range(0, Pl):

				# Determine individual scores
				if place == -1:
					# print("There was NO unicorn")
					if Players[k].decision == "Absent":
						# print("Player " + str(k) + "\'s answer is Correct!")
						Players[k].accuracy = True
						Players[k].score = Num_Loc*Num_Loc/2 - len(both)
						# print("Player " + str(k) + "\'s score this round is: " + \
						# 	str(Players[k].score))
					else:
						# print("Player " + str(k) + "\'s answer is Incorrect!")
						Players[k].accuracy = False
						Players[k].score = -Num_Loc*Num_Loc - len(both)
						# print("Player " + str(k) + "\'s score this round is: " + \
						# 	str(Players[k].score))
				else:
					# print("There was a unicorn")
					if Players[k].decision == "Present":
						# print("Player " + str(k) + "\'s answer is Correct!")
						Players[k].accuracy = True
						Players[k].score = Num_Loc*Num_Loc/2 - len(both)
						# print("Player " + str(k) + "\'s score this round is: " + \
						# 	str(Players[k].score))
					else:
						# print("Player " + str(k) + "\'s answer is Incorrect!")
						Players[k].accuracy = False
						Players[k].score = -Num_Loc*Num_Loc - len(both)
						# print("Player " + str(k) + "\'s score this round is: " + \
						# 	str(Players[k].score))

				row_of_data['Dyad'] = [dyad]
				row_of_data['Round'] = [i + 1]
				row_of_data['Player'] = [Players[k].name]
				row_of_data['Answer'] = [Players[k].decision]
				row_of_data['Time'] = [len(Players[k].where)]
				colA = ['a' + str(i+1) + str(j+1) for i in range(0, Num_Loc) for j in range(0, Num_Loc)]
				for l in range(0, Num_Loc * Num_Loc):
					if l in Players[k].where:
						row_of_data[colA[l]] = [1]
					else:
						row_of_data[colA[l]] = [0]
				row_of_data['Score'] = [Players[k].score]
				row_of_data['Joint'] = [len(both)]
				if place == -1:
					row_of_data['Is_there'] = ["Unicorn_Absent"]
					row_of_data['where_x'] = [-1]
					row_of_data['where_y'] = [-1]
				else:
					row_of_data['Is_there'] = ["Unicorn_Present"]
					x = place % Num_Loc
					y = (place - x) / Num_Loc
					row_of_data['where_x'] = [x]
					row_of_data['where_y'] = [y]

				row_of_data['Strategy'] = [Players[k].choice]

				# Add data to dataFrame
				dfAux = pd.DataFrame.from_dict(row_of_data)
				# print(dfAux)
				# print(dfAux.columns)
				# Keeping the order of columns
				dfAux = dfAux[['Dyad','Round','Player','Answer','Time','a11','a12','a13','a14','a15','a16','a17','a18','a21','a22','a23','a24','a25','a26','a27','a28','a31','a32','a33','a34','a35','a36','a37','a38','a41','a42','a43','a44','a45','a46','a47','a48','a51','a52','a53','a54','a55','a56','a57','a58','a61','a62','a63','a64','a65','a66','a67','a68','a71','a72','a73','a74','a75','a76','a77','a78','a81','a82','a83','a84','a85','a86','a87','a88','Score','Joint','Is_there','where_x','where_y','Strategy']]
				# print(dfAux)

				if TO_FILE:
				                with open('temp.csv', 'a') as f:
				                                dfAux.to_csv(f, header=False)
				else:
				                self.df = self.df.append(dfAux, ignore_index = True)

				# print(self.df)
				# print("Data from player " + str(k) + " has been saved")

				# Player determines its next strategy
				Players[k].joint = both

			if DEB:
				Is_there = " Absent" if place == -1 else " Present"
				titulo = "Round: " + str(i) + " (" + Is_there + ") Scores: (" + str(Players[0].score) + "," + str(Players[1].score)
				FRA.dibuja_ronda(Players[0], Players[1], titulo)

			for k in range(0, Pl):
				Players[k].make_decision()

			if DEB:
				print('-----------------')
				print('Unicorn ' + Is_there)
				# print('Region player 0:')
				# FRA.imprime_region(FRA.code2Vector(Players[0].where, Num_Loc))
				# print('Region player 1:')
				# FRA.imprime_region(FRA.code2Vector(Players[1].where, Num_Loc))
				print('both', len(both))
				print('scores: p0: ', Players[0].score, ' p1: ', Players[1].score)
				print('Player 0 to region ', FRA.nameRegion(Players[0].choice))
				print('Player 1 to region ', FRA.nameRegion(Players[1].choice))
				print('End summary round ', i)
				print('-----------------')

	def shake(self, strategy):
		if uniform(0, 1) > self.shaky_hand:
			p = 2
			outs = np.random.choice(strategy, p) if len(strategy) > 0 else []
			complement = [i for i in range(64) if i not in strategy]
			ins = np.random.choice(complement, p) if len(complement) > 0 else []
			return [i for i in strategy if i not in outs] + list(ins)
		else:
			return strategy

	def run_simulation(self):
		IT = self.gameParameters[4] # number of experiments in a set
		for h in range(0, IT):
			print("****************************")
			print("Running dyad no. ", h + 1)
			print("****************************\n")
			self.run_dyad()

	def save(self, archivo_raiz='./Data/output'):
		count = 0
		archivo = archivo_raiz + str(count) + '.csv'
		while os.path.isfile(archivo):
			count += 1
			archivo = archivo_raiz + str(count) + '.csv'
		self.df.to_csv(archivo, index=False)
		print('Data saved to' + archivo)
