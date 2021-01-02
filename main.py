# Simulation of probabilistic heuristic for WSLS and FRA solving "Seeking the unicorn" task
# Edgar Andrade-Lotero 2020
# Run with Python 3

print('Importing packages...')
import run_model as RM
print('Done!')


##########################################################################
#
#  Simulation starts here
#
##########################################################################

# Create experiment
p = 0.5 # probability of there being a unicorn
pl = 2 # number of players
n = 8 # number of rows/columns in grid
rounds = 60 # number of rounds
dyads = 5 # number of dyads
gameParameters = [p, pl, n, rounds, dyads]
# modelParameters = [0.14, 0.12, 0.07, 0.005, 0, 0, 0, 0, 0, 0]
# RM.standard_simulation(gameParameters, modelParameters, '5')
modelParameters = [0.1, 0.1, 0.1, 0.1, 500 ,1000, 30, 0, 0, 0]
RM.standard_simulation(gameParameters, modelParameters, '5')
# modelParameters = [0.25, 0.25, 0.12, 0, 500, 1000, 2, 8.36, 1000, 0.67]
# RM.standard_simulation(gameParameters, modelParameters, '5')
