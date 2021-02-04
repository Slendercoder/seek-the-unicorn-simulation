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

# # Simulations
dyads = 150 # number of dyads
gameParameters = [p, pl, n, rounds, dyads]
# modelParameters = [0.1, 0.1, 0.1, 0.1, 0, 0, 0, 0, 0, 0]
# RM.standard_simulation(gameParameters, modelParameters, '13')
# modelParameters = [0.1, 0.1, 0.1, 0.1, 100, 30, 31, 0, 0, 0]
# RM.standard_simulation(gameParameters, modelParameters, '13')
# modelParameters = [0.1, 0.1, 0.1, 0.1, 100, 30, 31, 2, 30, 0.8]
# RM.standard_simulation(gameParameters, modelParameters, '13')

# # Generating data for model recovery excercise
dyads = 50 # number of dyads
gameParameters = [p, pl, n, rounds, dyads]
# RM.data_for_confusion_matrix(gameParameters, N=100)

# With parameters from behavioral data
dyads = 45 # number of dyads
gameParameters = [p, pl, n, rounds, dyads]
modelParameters = [0.127, 0.076, 0.058, 0.005, 0, 0, 0, 0, 0, 0]
RM.standard_simulation(gameParameters, modelParameters, medidas='13', shaky_hand=0.88)
modelParameters = [0.1, 0.05, 0.018, 0.002, 38, 30, 4.6, 0, 0, 0]
RM.standard_simulation(gameParameters, modelParameters, medidas='13', shaky_hand=0.88)
modelParameters = [0.06, 0.05, 0.003, 0, 40, 30, 15, 0.52, 30, 0.95]
RM.standard_simulation(gameParameters, modelParameters, medidas='13', shaky_hand=0.88)
