# Measures are: accumulated score, normalized score, number of tiles visited,
# consistency, total of tiles visited dyad, difference in consistency, DLIndex,
# distance to closest focal path.

import numpy as np
import pandas as pd
import FRA

# --------------------------------------------------
# Global variables
# --------------------------------------------------
Num_Loc = 8
CLASIFICAR = False
CONTINUO = False
CONTADOR = 1
TOLERANCIA = 4

# List of columns for region visited
cols1 = ['a' + str(i) + str(j) \
for i in range(1, Num_Loc + 1) \
for j in range(1, Num_Loc + 1) \
]

regionsCod = ['abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
                  '', # NOTHING
                  'GHIJKLMNOPQRSTUVWXYZ0123456789;:', # BOTTOM
                  'abcdefghijklmnopqrstuvwxyzABCDEF', # TOP
                  'abcdijklqrstyzABGHIJOPQRWXYZ4567', # LEFT
                  'efghmnopuvwxCDEFKLMNSTUV012389;:', # RIGHT
                  'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012', # IN
                  'abcdefghipqxyFGNOVW3456789;:' # OUT
                  ]

regionsCoded = [FRA.code2Vector(FRA.lettercode2Strategy(x, Num_Loc), Num_Loc) for x in regionsCod]

# --------------------------------------------------
# Functions
# --------------------------------------------------

def obtainPresentBlocks(x):

    global CONTADOR

    valor = CONTADOR

    if x['Is_there'] == 'Unicorn_Present' and x['Is_there_LEAD'] == 'Unicorn_Absent':
        CONTADOR += 1

    if pd.isna(x['Is_there_LEAD']):
        CONTADOR += 1

    if x['Is_there'] == 'Unicorn_Present':
        return valor
    else:
        return 0

def obtainIndicesIncluir(x):

    if x['Cambio_LAG1'] == 0 and x['Cambio'] != 0:
        return 1
    else:
        return 0

def correctavSc(x):
    if x['Is_there'] == 'Unicorn_Present':
        return np.ceil(x['avScGrpUniPresent'])
    else:
        return x['Score']


def nextScore(si, siLead, s, sLEAD):
    if si == 'Unicorn_Absent' and siLead == 'Unicorn_Present' and s > 29 and s > sLEAD:
        return sLEAD
    else:
        return s

# def calcula_consistencia(x, y):
#     joint = np.multiply(x,y)
#     total_visited = np.add(x,y)
#     total_visited = total_visited.astype(float)
#     total_visited = total_visited * 0.5
#     total_visited = np.ceil(total_visited)
#     j = np.sum(joint)
#     t = np.sum(total_visited)
#     if t != 0:
#         return j/t
#     else:
#         return 1

# Function to insert row in the dataframe
def Insert_row(row_number, df, row_value):
    # Starting value of upper half
    start_upper = 0

    # End value of upper half
    end_upper = row_number

    # Start value of lower half
    start_lower = row_number

    # End value of lower half
    end_lower = df.shape[0]

    # Create a list of upper_half index
    upper_half = [*range(start_upper, end_upper, 1)]

    # Create a list of lower_half index
    lower_half = [*range(start_lower, end_lower, 1)]

    # Increment the value of lower half by 1
    lower_half = [x.__add__(1) for x in lower_half]

    # Combine the two lists
    index_ = upper_half + lower_half

    # Update the index of the dataframe
    df.index = index_

    # Insert a row at the end
    df.loc[row_number] = row_value

    # Sort the index labels
    df = df.sort_index()

    # return the dataframe
    return df

def completeRegions(strat, columna):

    if strat == 0 or strat == 9:
        v = FRA.code2Vector(FRA.new_random_strategy(Num_Loc), Num_Loc)
    else:
        v = regionsCoded[strat - 1] # -1 since regionsCoded does not have RS

    i = cols1.index(columna)
    return v[i]

def deleteScore(x):
    if x['Is_there'] == 'Unicorn_Present':
        return np.nan
    else:
        return x['Score']

def get_measures(data, lista):

    '''List of measures:
    0: Complete from simulation (for simulated data only)
    1: Classify regions
    2: Correct scores
    3: Estimate blocks
    4: Keep only absent
    5: Find max similarity'''

    global cols1

    print(data.head())
    print("Sorting by Dyad, Player, Round...")
    data = data.sort_values(['Dyad', 'Player', 'Round'], ascending=[True, True, True]).reset_index(drop=True)
    # data.to_csv('output_Prev.csv', index=False)
    data['Is_there_LEAD'] = data.groupby(['Dyad', 'Player'])['Is_there'].transform('shift', periods=-1)

    # --------------------------------------------------
    # Obtaining measures from players' performance
    # --------------------------------------------------
    # Making sure Score is an integer
    data['Score'] = data['Score'].apply(int)
    # Find the accumulated score
    print("Finding accumulated score...")
    data['Ac_Score'] = data.sort_values(['Dyad','Player']).groupby('Player')['Score'].cumsum()
    # Find normalized score
    print("Finding normalized score...")
    max_score = 32
    min_score = -64 - 64
    data['Norm_Score'] = (data['Score'] - min_score) / (max_score - min_score)
    # print data
    print("Finding the initial lag variables...")
    data['Joint_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Joint'].transform('shift', 1)

    if '0' in lista:
        # --------------------------------------------------
        # Completing from simulation
        # --------------------------------------------------
        print("Completing regions from simulation (please be patient!)...")
        for c in cols1:
            # print('Working with column', c)
            data[c] = data.apply(lambda x: completeRegions(x['Strategy'], c), axis=1)

    if '2' in lista:
        # 1. Create column of indexes
        data = data.reset_index()
        data['indice'] = data.index

        # 2. Indices de comienzo de jugador
        indiceJugador = list(data.groupby('Player')['indice'].first())
        indiceJugador.sort()
        # print('indiceJugador', indiceJugador)

        # 2. Obtain indices of blocks of Unicorn_Present
        data['Cambio'] = data.apply(obtainPresentBlocks, axis=1)
        # print('List of blocks\n', data[['Player', 'Round', 'Is_there', 'Cambio']][50:60])

        # 3. Obtain average score per group of Unicorn_Present
        data['avScGrpUniPresent'] = data.groupby('Cambio')['Score'].transform('mean')
        data['avScGrpUniPresent'] = data.apply(correctavSc, axis=1)
        data['avScGrpUniPresent_LEAD'] = data.groupby(['Dyad', 'Player'])['avScGrpUniPresent'].transform('shift', -1)
        # print('List of blocks\n', data[['Player', 'Is_there', 'Score', 'avScGrpUniPresent']][50:60])

        # --------------------------------------------------
        # Correcting scores
        # --------------------------------------------------
        print('Correcting scores...')
        # 4. Correct score from last round absent to average score next block present
        data['Score'] = data.apply(lambda x: nextScore(x['Is_there'], x['Is_there_LEAD'], x['Score'], x['avScGrpUniPresent_LEAD']), axis=1)
        # print('List of blocks\n', data[['indice', 'Player', 'Round', 'Is_there', 'Score', 'Category']][50:60])

    if '3' in lista:

        CLASIFICAR = False
        CONTINUO = False
        CONTADOR = 1

        # 1. Create column of indexes
        data = data.reset_index()
        data['indice'] = data.index

        # 2. Indices de comienzo de jugador
        indiceJugador = list(data.groupby('Player')['indice'].first())
        indiceJugador.sort()
        # print('indiceJugador', indiceJugador)

        # 2. Obtain indices of blocks of Unicorn_Present
        data['Cambio'] = data.apply(obtainPresentBlocks, axis=1)
        # print('List of blocks\n', data[['Player', 'Round', 'Is_there', 'Cambio']][0:20])

        # 3. Obtain average score per group of Unicorn_Present
        data['avScGrpUniPresent'] = data.groupby('Cambio')['Score'].transform('mean')
        data['avScGrpUniPresent'] = data.apply(correctavSc, axis=1)
        data['avScGrpUniPresent_LEAD'] = data.groupby(['Dyad', 'Player'])['avScGrpUniPresent'].transform('shift', -1)
        # print('List of blocks\n', data[['Player', 'Is_there', 'Score', 'avScGrpUniPresent']][0:20])

        # --------------------------------------------------
        # Estimating blocks
        # --------------------------------------------------
        print('Estimating blocks (please be patient)...')

        # 5. Obtain indicesIncluir
        data['Cambio_LAG1'] = data.groupby(['Dyad', 'Player'])['Cambio'].transform('shift', periods=1)
        data['Aux1'] = data.apply(lambda x: obtainIndicesIncluir(x), axis=1)
        # print('List of blocks\n', data[['indice', 'Player', 'Is_there', 'Cambio', 'Cambio_LAG1', 'Aux1']][50:60])
        # print('List of blocks\n', data[['Round', 'Player', 'Is_there', 'Cambio', 'Cambio_LAG1', 'Aux1']][110:])
        indicesIncluir = data.indice[data['Aux1'] == 1].tolist()
        data = data.drop(columns=['Aux1', 'Cambio_LAG1'])
        # print('indicesIncluir', indicesIncluir)

        # 6. Include new row of Unicorn_Absent with estimated region and previous score
        for k in range(len(indicesIncluir)):
            Ind = len(indicesIncluir) - k - 1
            c = indicesIncluir[Ind]
            if c not in indiceJugador:
                valor = data.loc[c]['Cambio']
                # print('Processing estimation (c)', c, 'with Cambio', valor)
                df_aux = data[data['Cambio'] == valor]
                proxInd = list(df_aux.indice)[-1] + 1
                # print('Block to be estimated\n', df_aux[['indice', 'Round', 'Player', 'Is_there', 'Category', 'Score']])
                columnas = df_aux.columns
                # print(columnas)
                columnas_no = ['Dyad', 'Round', 'Player', 'Answer', 'Is_there', 'where_x', 'where_y', 'Strategy', 'Is_there_LEAD', 'Category', 'indice', 'Cambio', 'puntaje']
                columnas_no += cols1
                dict_aux = {}
                dict_aux['Dyad'] = data.loc[c]['Dyad']
                dict_aux['Round'] = data.loc[c]['Round']
                dict_aux['Player'] = data.loc[c]['Player']
                dict_aux['Answer'] = 'Absent'
                dict_aux['Is_there'] = 'Unicorn_Absent'
                dict_aux['where_x'] = -1
                dict_aux['where_y'] = -1
                try:
                    dict_aux['Is_there_LEAD'] = data.loc[c+1]['Is_there_LEAD']
                except:
                    dict_aux['Is_there_LEAD'] = np.nan
                dict_aux['indice'] = c
                dict_aux['Cambio'] = 0
                puntaje = np.ceil(df_aux['Score'].mean())
                dict_aux['Score'] = puntaje
                if puntaje > 31:
                    v = list(data.loc[proxInd][cols1])
                    Category_v = data.loc[proxInd]['Category']
                else:
                    v = FRA.code2Vector(FRA.new_random_strategy(Num_Loc), Num_Loc)
                    Category_v = 'RS'

                for x in range(len(cols1)):
                    dict_aux[cols1[x]] = v[x]

                for co in columnas:
                    if co not in columnas_no:
                        auxi = np.ceil(df_aux[co].mean())
                        dict_aux[co] = auxi

                dict_aux['Category'] = Category_v
                dict_aux['Strategy'] = FRA.numberRegion(Category_v)
                # print('Dict\n', dict_aux)
                row_number = c
                row_value = [dict_aux[x] for x in columnas]
                # print("Estimation:\n", row_value)
                data = Insert_row(row_number, data, row_value)
            # print('Estimated row\n', data[['indice', 'Round', 'Player', 'Is_there', 'Category', 'Score']][row_number-1:row_number+2])

        # Must correct scores again
        CLASIFICAR = False
        CONTINUO = False
        CONTADOR = 1

        # 1. Create column of indexes
        # data = data.reset_index()
        data['indice'] = data.index

        # 2. Indices de comienzo de jugador
        indiceJugador = list(data.groupby('Player')['indice'].first())
        indiceJugador.sort()
        # print('indiceJugador', indiceJugador)

        # 2. Obtain indices of blocks of Unicorn_Present
        data['Cambio'] = data.apply(obtainPresentBlocks, axis=1)
        # print('List of blocks\n', data[['Player', 'Round', 'Is_there', 'Cambio']][50:60])

        # 3. Obtain average score per group of Unicorn_Present
        data['avScGrpUniPresent'] = data.groupby('Cambio')['Score'].transform('mean')
        data['avScGrpUniPresent'] = data.apply(correctavSc, axis=1)
        data['avScGrpUniPresent_LEAD'] = data.groupby(['Dyad', 'Player'])['avScGrpUniPresent'].transform('shift', -1)
        # print('List of blocks\n', data[['Player', 'Is_there', 'Score', 'avScGrpUniPresent']][50:60])

        # --------------------------------------------------
        # Correcting scores
        # --------------------------------------------------
        print('Correcting scores...')
        # 4. Correct score from last round absent to average score next block present
        data['Score'] = data.apply(lambda x: nextScore(x['Is_there'], x['Is_there_LEAD'], x['Score'], x['avScGrpUniPresent_LEAD']), axis=1)
        # print('List of blocks\n', data[['indice', 'Player', 'Round', 'Is_there', 'Score', 'Category']][50:60])

    if '4' in lista:
        # 1. Create column of indexes
        data = data.reset_index()
        data['indice'] = data.index

        # 2. Indices de comienzo de jugador
        indiceJugador = list(data.groupby('Player')['indice'].first())
        indiceJugador.sort()
        # print('indiceJugador', indiceJugador)

        # 2. Obtain indices of blocks of Unicorn_Present
        data['Cambio'] = data.apply(obtainPresentBlocks, axis=1)
        # print('List of blocks\n', data[['Player', 'Round', 'Is_there', 'Cambio']][0:20])

        # 7. Keep only rounds with Unicorn_Absent
        print('Keeping only rounds with Unicorn Absent...')
        # Obtaining nans from unicorn_present blocks
        # data['Score'] = data.groupby('Cambio').transform(deleteScore)
        data['Score'] = data.apply(deleteScore, axis=1)
        data['ScoreLEAD'] = data.groupby(['Dyad', 'Player'])\
                                    ['Score'].transform('shift', -1)
        data = pd.DataFrame(data.groupby('Is_there').get_group('Unicorn_Absent'))
        print('List of blocks\n', data[['Round', 'Is_there', 'Score', 'ScoreLEAD']][0:20])

    if '1' in lista:
        # --------------------------------------------------
        # Classify region per round, per player
        # --------------------------------------------------
        print("Classifying regions...")

        # Deterimining list of columns
        cols1 = ['a' + str(i) + str(j) for i in range(1, Num_Loc + 1) for j in range(1, Num_Loc + 1)]
        data['Category'] = data.apply(lambda x: FRA.classify_region(x[cols1], regionsCoded, TOLERANCIA), axis=1)

    else:
        print("Trying to obtain classification from simulation...")
        try:
            data['Category'] = data.apply(lambda x: FRA.nameRegion(x['Strategy']), axis=1)
            # print('Done!')
        except:
            print('Data does not seem to come from simulation!')

    # --------------------------------------------------
    # Obtaining final measures from players' performance
    # --------------------------------------------------
    # Find Size_visited
    print("Finding Size_visited...")
    # print('cols: ', cols1)
    data['Size_visited'] = data[cols1].sum(axis=1)
    # print(data[['Player', 'Round', 'Size_visited', 'Joint']][:10])
    # assert(all(data['Size_visited'] >= data['Joint']))
    # #
    # print("Sorting by Player...")
    # data = data.sort_values(['Player', 'Round'], \
    #                 ascending=[True, True])
    #
    # Find consistency
    print("Finding consistency...")
    # # print data[:10]
    data['Vector'] = data.apply(lambda x: np.array(x[cols1]), axis=1)
    data['VectorLAG1'] = data.groupby(['Dyad', 'Player'])['Vector'].transform('shift', 1)
    # data = data.dropna()
    data['Consistency'] = data.apply(lambda x: FRA.sim_consist(x['Vector'], x['VectorLAG1']), axis=1)
    del data['VectorLAG1']

    # Find difference in consistency, Total_visited_dyad, and reparing Joint ----
    print("Finding difference in consistency and Total_visited_dyad...")
    cols = ['Dyad','Player','Consistency','Round','Joint', 'Size_visited', 'Vector']
    total = {}
    dif_cons = {}
    joints = {}
    for key, grp in data[cols].groupby(['Dyad']):
    	Players = grp.Player.unique()
    	# print "The players in dyad " + str(key) + " are: " + str(Players)
    	Grp_player = grp.groupby(['Player'])
    	aux1 = pd.DataFrame(Grp_player.get_group(Players[0])).reset_index()
    	# print "aux1: \n", aux1
    	aux2 = pd.DataFrame(Grp_player.get_group(Players[1])).reset_index()
    	# print "aux2: \n", aux2
    	# print "len(aux1)", len(aux1)
    	if len(aux1) != len(aux2):
    	       print("Oops, something went wrong with estimated blocks!")
    	       print("Trying to correct...")
    	       print(list(aux1.Round))
    	       d = aux1[aux1['Round'] == 60].index
    	       print('Index round 60', d)
    	       print(aux1.loc[d]['Round'])
    	       aux1 = aux1.drop(d)
    	       print(list(aux1.Round))

    	errorDebug = "Something wrong with players!\n"
    	errorDebug += "Dyad " + str(key)
    	errorDebug += "\nlen player 1 " + str(len(aux1))
    	errorDebug += "\nlen player 2 " + str(len(aux2))
    	errorDebug += "\nRounds player 1 " + str(list(aux1.Round))
    	errorDebug += "\nRounds player 1 " + str(list(aux2.Round))
    	assert(len(aux1) == len(aux2)), errorDebug
    	aux3 = pd.DataFrame({'Dyad':aux1['Dyad'],\
    	'Round':aux1['Round'],\
    	'C1':aux1['Consistency'],\
    	'C2':aux2['Consistency'],\
    	'v1':aux1['Vector'],\
    	'v2':aux2['Vector'],\
    	'V1':aux1['Size_visited'],\
    	'V2':aux2['Size_visited']})
    	aux3['Joint'] = aux3.apply(lambda x: np.sum(np.multiply(x['v1'],x['v2'])), axis=1)
    	aux3['total_visited'] = aux3.apply(lambda x: x['V1'] + x['V2'] - x['Joint'], axis=1)
    	aux3['Dif_consist'] = aux3.apply(lambda x: np.abs(x['C1'] - x['C2']), axis=1)
    	aux3['Pair'] = aux3.apply(lambda x: tuple([x['Dyad'], x['Round']]), axis=1)
    	joints1 = dict(zip(aux3.Pair, aux3.Joint))
    	joints = {**joints, **joints1}
    	total1 = dict(zip(aux3.Pair, aux3.total_visited))
    	total = {**total, **total1}
    	dif_cons1 = dict(zip(aux3.Pair, aux3.Dif_consist))
    	dif_cons = {**dif_cons, **dif_cons1}

    data['Pair'] = data.apply(lambda x: tuple([x['Dyad'], x['Round']]), axis=1)
    data['Joint'] = data['Pair'].map(joints)
    data['Total_visited_dyad'] = data['Pair'].map(total)
    data['Dif_consist'] = data['Pair'].map(dif_cons)
    del data['Vector']
    del data['Pair']

    # Division of labor Index
    data['DLIndex'] = (data['Total_visited_dyad'] - data['Joint'])/(Num_Loc*Num_Loc)
    assert(all(data['DLIndex'] >= 0)), str(list(data.loc[data['DLIndex']==0].index))

    if '5' in lista:
        # --------------------------------------------------
        # Finding distance to closest focal region per round, per player
        # --------------------------------------------------
        print("Finding distances to focal regions (please be patient)...")
        # data['Similarity'] = data.apply(lambda x: FRA.maxSim2Focal(x[cols1], Num_Loc), axis=1)
        data['Similarity'] = data.apply(lambda x: FRA.minDist2Focal(x[cols1], regionsCoded), axis=1)

    # --------------------------------------------------
    # Finding the lag and lead variables
    # --------------------------------------------------
    print("Finding the last lag variables...")
    data['Score_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Score'].transform('shift', 1)
    data['Norm_Score_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Norm_Score'].transform('shift', 1)
    data['Consistency_LEAD1'] = data.groupby(['Dyad', 'Player'])\
                                ['Consistency'].transform('shift', -1)
    data['Dif_consist_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Dif_consist'].transform('shift', 1)
    data['Category_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Category'].transform('shift', 1)
    data['RegionGo'] = data.groupby(['Dyad', 'Player'])\
                                ['Category'].transform('shift', -1)
    data['RegionGo2'] = data.groupby(['Dyad', 'Player'])\
                                ['Category'].transform('shift', -2)
    if '5' in lista:
        data['Similarity_LAG1'] = data.groupby(['Dyad', 'Player'])\
                                ['Similarity'].transform('shift', 1)

    if '4' in lista:
        data = pd.DataFrame(data.groupby('Is_there').get_group('Unicorn_Absent'))#.reset_index()
        data = data[data['ScoreLEAD'].notna()]
        print('List of blocks\n', data[['Round', 'Is_there', 'Score', 'ScoreLEAD']][0:20])


    return data
