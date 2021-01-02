library(dplyr)
library(dfoptim)
library(bbmle)

###########################
# Global variables
###########################

letras <- 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:'
letras <- strsplit(letras, split = '')
letras <- letras[[1]]

regiones <- c('RS',
              'ALL', 
              'NOTHING', 
              'BOTTOM', 
              'TOP', 
              'LEFT', 
              'RIGHT', 
              'IN', 
              'OUT')

regionsCoded <- c('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789;:', # ALL
                  '', # NOTHING
                  'GHIJKLMNOPQRSTUVWXYZ0123456789;:', # BOTTOM
                  'abcdefghijklmnopqrstuvwxyzABCDEF', # TOP
                  'abcdijklqrstyzABGHIJOPQRWXYZ4567', # LEFT
                  'efghmnopuvwxCDEFKLMNSTUV012389;:', # RIGHT
                  'jklmnorstuvwzABCDEHIJKLMPQRSTUXYZ012', # IN
                  'abcdefghipqxyFGNOVW3456789;:') # OUT


lowerEps2=.00001
highEps2 =.99999

#lower_limits=c(0,0,0,0,0,0,0,0,0,0)
#upper_limits=c(0.1,0.1,0.1,0.1,500,1000,32,500,1000,2)

lower_limits=c(0,0,0,0,0,499,0,0,499,0)
upper_limits=c(0.1,0.1,0.1,0.1,500,500,32,500,500,2)

###########################
# Define functions
###########################

specify_decimal3 <- function(x) trimws(format(round(x, 3), nsmall=3))
imprimir <- function(x) print(as.numeric(unlist(lapply(x, specify_decimal3))))
para_visualizar <- function(theta) {
  a <- as.character(theta)
  params <- ''
  for (letra in a) {
    params <- paste(params, letra, ', ', sep = "")
  }
  return(params)
}

letterCode <- function(x, letras) {
  code <- ''
  for (i in 1:length(x)) {
    if (x[i]==1) {
      code <- paste(code, letras[i], sep = '')
    }
  }
  #  if (code=='') {code <- 'NOTHING'}
  return(code)
}

classifyCode <- function(cadena) {
  # Returns the code of a given lettercode
  # Input: cadena,  lettercode of a region
  # Output: name of the region
  
  index <- which(regionsCoded == cadena)
  #  print(index)
  if (length(index) == 0) {
    return ('RS')
  } else {
    return(regiones[index + 1])
  }
} # end classifyCode

code2Vector <- function(cadena) {
  # Returns the vector of 0s and 1s from a given code
  # Input: cadena, a string with the code in characters
  # Output: vector of 0s and 1s of lenght 64
  
  v <- rep(0, length(letras))
  cadena <- strsplit(cadena, split = '')
  cadena <- cadena[[1]]
  for (i in 1:length(cadena)) {
    index <- which(letras == cadena[i])[1]
    v[index] <- 1
  }
  return(v)
} # end code2Vector

specify_decimal3 <- function(x) trimws(format(round(x, 3), nsmall=3))
imprimir <- function(x) print(as.numeric(unlist(lapply(x, specify_decimal3))))

sigmoid <- function(x, beta, gamma) {
  # Returns the value of the sigmoid function 1/(1+exp(b(x-c)))
  
  return(1 / (1 + exp(-beta * (x - gamma))))
} # end sigmoid

find_regionVector <- function(df) {
  
  #  df <- df[complete.cases(df), ]
  
  # Create vector for columns with region
  columns <- c()
  for (i in c(1:8)) {
    for (j in c(1:8)) {
      columns <- append(columns, paste('a', paste(i, j, sep=''), sep=''))
    }
  }
  
  # Build the region column per player
  df$vR <- lapply(as.list(as.data.frame(t(df[columns]))), function(x) x)
  
  return (df)
  
}

find_joint_region <- function(df1) {
  
  # Requires the visited regions as vectors
  df1 <- find_regionVector(df1)
  
  # Must optimize this procedure with dplyr!
  # Initialize auxiliary dataframe
  auxDF <- data.frame(c('Dyad', NA), 
                      c('Player', NA), 
                      c('RegionFULL', NA), 
                      c('RJoint', NA), 
                      c('Score', NA))
  colnames(auxDF) = as.character(unlist(auxDF[1, ])) # the first row will be the header
  auxDF = auxDF[-1, ]          # removing the first row.
  auxDF = auxDF[-1, ]          # removing the first row.
  #auxDF$RJoint <- list(0)
  #auxDF$Score <- 0
  
  parejas <- unique(df1$Dyad)
  pareja <- parejas[2]
  
  for (pareja in unique(df1$Dyad)) {
    # Create the joint region
    parejaDF <- df1[which(df1$Dyad == pareja), ]
    parejaDF[order(parejaDF$Round), ]
    jugador <- unique(parejaDF$Player)
    r1 <- parejaDF$vR[which(parejaDF$Player == jugador[1])]
    r2 <- parejaDF$vR[which(parejaDF$Player == jugador[2])]
    newDF <- data.frame(rep(0, length(r1)))
    newDF$a <- r1
    newDF$b <- r2
    lst <- as.list(as.data.frame(t(newDF)))
    newDF$rJoint <- lapply(lst, function(x) as.numeric(unlist(x[2])) * as.numeric(unlist(x[3])))
    
    # Create dataframe for first player
    DF <- data.frame(seq(1, length(r1), by=1))
    DF$Dyad <- rep(pareja, length(r1))
    DF$Player <- rep(as.character(jugador[1]), length(r1))
    DF$Region <- parejaDF$Category[which(parejaDF$Player == jugador[1])]
    DF$RegionFULL <- parejaDF$vR[which(parejaDF$Player == jugador[1])]
    DF$RegionGo <- lead(DF$Region, 1)
    DF$RJoint <- newDF$rJoint
    DF$Score <- parejaDF$Score[which(parejaDF$Player == jugador[1])]
    DF <- DF[c('Dyad', 'Player', 'Region', 'RegionFULL', 'RegionGo', 'RJoint', 'Score')]
    #  DF <- DF[c('Dyad', 'Player', 'RegionFULL', 'RJoint', 'Score')]
    
    # Add dataframe to big dataframe
    auxDF <- rbind(auxDF, DF)
    #  auxDF <- na.omit(auxDF)
    
    # Create dataframe for second player
    DF <- data.frame(seq(1, length(r2), by=1))
    DF$Dyad <- rep(pareja, length(r2))
    DF$Player <- rep(as.character(jugador[2]), length(r2))
    DF$Region <- parejaDF$Category[which(parejaDF$Player == jugador[2])]
    DF$RegionFULL <- parejaDF$vR[which(parejaDF$Player == jugador[2])]
    DF$RegionGo <- lead(DF$Region, 1)
    DF$RJoint <- newDF$rJoint
    DF$Score <- parejaDF$Score[which(parejaDF$Player == jugador[2])]
    DF <- DF[c('Dyad', 'Player', 'Region', 'RegionFULL', 'RegionGo', 'RJoint', 'Score')]
    #  DF <- DF[c('Dyad', 'Player', 'RegionFULL', 'RJoint', 'Score')]
    
    # Add dataframe to big dataframe
    auxDF <- rbind(auxDF, DF)
    #  auxDF <- na.omit(auxDF)
  }
  head(auxDF)
  dim(auxDF)
  auxDF$Player <- as.character(auxDF$Player)
  
  # Code RegionFULL
  lst <- as.list(as.data.frame(t(auxDF$RegionFULL)))
  regionesJuntos <- lapply(lst, function(x) {
    cadena <- as.character(unlist(x))
    letterCode(cadena, letras)
  })
  auxDF$RegionFULL <- regionesJuntos
  
  # Code overlapping regions 
  lst <- as.list(as.data.frame(t(auxDF$RJoint)))
  regionesJuntos <- lapply(lst, function(x) {
    cadena <- as.character(unlist(x))
    letterCode(cadena, letras)
  })
  auxDF$RJcode <- regionesJuntos
  
  aux <- auxDF[c('Dyad', 'Player', 'Region', 'RegionFULL', 'RegionGo', 'RJcode', 'Score')]
  aux$Region <- as.character(aux$Region)
  aux$RegionGo <- as.character(aux$RegionGo)
  #  aux$Rcode <- as.character(aux$Rcode)
  aux$RJcode <- as.character(aux$RJcode)
  head(aux)
  
  return(aux)
  
}

obtainFreqVector <- function(x) {
  a <- data.frame(table(x))
  return(list(a$Freq))
}

regNoCod <- function(x) {
  index1 <- which(df$RegionFULL == x)
  b <- unique(df$Region[index1])
  if (length(b) > 1) {
    msg <- paste('Oops!', l, index1)
    print(msg)
  }
  return(b)
}

getFreqFRA <- function(df, theta) {
  
  df <- df[c('RegionFULL', 'Score', 'RJcode', 'RegionGo')]
  df$RegionGo <- factor(df$RegionGo, levels = regiones)
  dfA <- df %>%
    dplyr::group_by(RegionFULL, Score, RJcode) %>%
    dplyr::summarize(Freqs = obtainFreqVector(RegionGo))
  
  dfA$Region <- lapply(dfA$RegionFULL, regNoCod)
  dfA$Region <- unlist(dfA$Region)
  
  dfA$freqs <- lapply(dfA$Freqs, function(x) {
    x1 <- x[1]
    x2 <- x[2]
    x3 <- x[3]
    x4 <- x[4]
    x5 <- x[5]
    x6 <- x[6]
    x7 <- x[7]
    x8 <- x[8]
    x9 <- x[9]
    return (c(x1, x2, x3, x4, x5, x6, x7, x8, x9))
  })
  
  return (dfA[c('Region', 'RegionFULL', 'Score', 'RJcode', 'freqs')])
} 

sim_consist <- function(v1, v2){
  # Returns the similarity based on consistency
  # v1 and v2 are two 64-bit coded regions
  
  joint <- v1 * v2
  union <- (v1 + v2) / (v1 + v2)
  union[is.na(union)] <- 0
  j = sum(joint)
  u = sum(union)
  if (u != 0) {
    return (j/u)
  } else {
    return (1)
  }
} # end sim_consist

FRAsim <- function(i, iV, j, k) {
  
  # Returns the FRAsim of i and j with respect to focal k
  #Input: i, the name of the region the player is in
  #        iV (lettercode), the region the player is in
  #        s, the player's score on the round
  #        j  (lettercode), the overlapping region
  #        k  (lettercode), a focal region
  
  index1 <- which(regiones == k)
  kCoded <- regionsCoded[index1 - 1] # regionsCoded does not have 'RS'
  kV <- code2Vector(kCoded)
  
  simil <- sim_consist(code2Vector(iV), kV)
  
  #  if (k!='ALL' && k!='NOTHING') {
  if (k!='ALL') {
    kVComp <- 1 - code2Vector(kCoded)
    simil2 <- sim_consist(code2Vector(j), kVComp)
    simil <- simil + simil2
  }
  
  return (simil)
  
} # End FRAsim

get_FRASims <- function(df) {
  
  df$FRASimALL <- mapply(function(i, iv, j) FRAsim(i, iv, j,'ALL'),
                         df$Region,
                         df$RegionFULL,
                         df$RJcode)
  
  df$FRASimNOTHING <- mapply(function(i, iv, j) FRAsim(i, iv, j,'NOTHING'),
                             df$Region,
                             df$RegionFULL,
                             df$RJcode)
  
  df$FRASimBOTTOM <- mapply(function(i, iv, j) FRAsim(i, iv, j,'BOTTOM'),
                          df$Region,
                          df$RegionFULL,
                          df$RJcode)
  
  df$FRASimTOP <- mapply(function(i, iv, j) FRAsim(i, iv, j,'TOP'),
                        df$Region,
                        df$RegionFULL,
                        df$RJcode)
  
  df$FRASimLEFT <- mapply(function(i, iv, j) FRAsim(i, iv, j,'LEFT'),
                          df$Region,
                          df$RegionFULL,
                          df$RJcode)
  
  df$FRASimRIGHT <- mapply(function(i, iv, j) FRAsim(i, iv, j,'RIGHT'),
                           df$Region,
                           df$RegionFULL,
                           df$RJcode)
  
  df$FRASimIN <- mapply(function(i, iv, j) FRAsim(i, iv, j,'IN'),
                        df$Region,
                        df$RegionFULL,
                        df$RJcode)
  
  df$FRASimOUT <- mapply(function(i, iv, j) FRAsim(i, iv, j,'OUT'),
                         df$Region,
                         df$RegionFULL,
                         df$RJcode)
  
  return (df)
  
} # end get_FRASims

get_FRASims_list <- function(df) {
  
  aux <- get_FRASims(args)
  
  aux$FRASims <- mapply(function(x1,x2,x3,x4,x5,x6,x7,x8) as.list(data.frame(c(x1,x2,x3,x4,x5,x6,x7,x8))),
                        aux$FRASimALL, 
                        aux$FRASimNOTHING, 
                        aux$FRASimBOTTOM, 
                        aux$FRASimTOP, 
                        aux$FRASimLEFT, 
                        aux$FRASimRIGHT, 
                        aux$FRASimIN, 
                        aux$FRASimOUT
  )
  
  # dplyr is creating problems: it returns lists longer than expected for some rows 
  #  aux <- aux %>% 
  #    dplyr::mutate(FRASims = as.list(data.frame(c(FRASimALL, 
  #                                                 FRASimNOTHING, 
  #                                                 FRASimBOTTOM, 
  #                                                 FRASimTOP, 
  #                                                 FRASimLEFT, 
  #                                                 FRASimRIGHT, 
  #                                                 FRASimIN, 
  #                                                 FRASimOUT)))
  #    ) %>%
  
  aux <- aux %>% select(Region, RegionFULL, Score, RJcode, FRASims, freqs)
  
  return(aux)
  
} # end get_FRASims_list


getFreq_based_on_FRASim <- function(df, k) {
  
  if (k == 'ALL') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimALL', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimALL, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimALL) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimALL
    
  } # end k='ALL'
  
  if (k == 'NOTHING') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimNOTHING', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimNOTHING, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimNOTHING) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimNOTHING
    
  } # end k='NOTHING'
  
  if (k == 'BOTTOM') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimBOTTOM', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimBOTTOM, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimBOTTOM) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimBOTTOM
    
  } # end k='BOTTOM'
  
  if (k == 'TOP') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimTOP', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimTOP, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimTOP) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimTOP
    
  } # end k='TOP'
  
  if (k == 'LEFT') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimLEFT', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimLEFT, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimLEFT) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimLEFT
    
  } # end k='LEFT'
  
  if (k == 'RIGHT') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimRIGHT', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimRIGHT, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimRIGHT) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimRIGHT
    
  } # end k='RIGHT'
  
  if (k == 'IN') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimIN', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimIN, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimIN) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimIN
    
  } # end k='IN'
  
  if (k == 'OUT') {
    df <- df[complete.cases(df$RegionGo), ]
    df <- df[df$RegionGo != "", ]
    df <- df %>% select('Region', 'FRASimOUT', 'RegionGo')
    df <- df %>%
      dplyr::group_by(Region, FRASimOUT, RegionGo) %>%
      dplyr::summarize(n = n()) %>%
      dplyr::ungroup() %>%
      dplyr::group_by(Region, FRASimOUT) %>%
      dplyr::mutate(n1 = sum(n),
                    Freqs = n/n1)  
    
    df$FRASim <- df$FRASimOUT
    
  } # end k='OUT'
  
  return(df[c('Region', 'FRASim', 'RegionGo', 'Freqs')])
  
} # End get_freq_based_on_FRAsim

FRApred <- function(i, iV, s, j, theta) {
  
  wAll <- theta[1]
  wNoth <- theta[2]
  wLef <- theta[3]
  wIn <- theta[4]
  alpha <- theta[5]
  beta <- theta[6]
  gamma <- theta[7]
  delta <- theta[8]
  epsilon <- theta[9]
  zeta <- theta[10]
  
  # First we calculate the prior probabilities
  aux <- c(wAll, wNoth, wLef, wLef, wLef, wLef, wIn, wIn)
  # The probability of region 'RS' is 1 - the sum of the other probabilities
  if (sum(aux) > 1) {
    aux <- aux/sum(aux)
  }
  bias <- c(1 - sum(aux), aux)
  #  imprimir(bias)
  
  # Start from biases
  attractiveness <- bias
  # Add WinStay
  index <- which(regiones == i)
  # adding win stay only to focal regions
  if (i != 'RS') {
    attractiveness[index] <- attractiveness[index] + alpha * sigmoid(s, beta, gamma) 
  }
  #  print('Attractiveness with WS:')
  #  imprimir(attractiveness)
  
  similarities <- lapply(regiones[2:9], function(x) {
    f <- FRAsim(i, iV, j, x) 
    return(delta * sigmoid(f, epsilon, zeta))
  })
  similarities <- c(0, unlist(similarities))
  attractiveness <- attractiveness + similarities
  
  attractiveness <- replace(attractiveness,attractiveness<lowerEps2,lowerEps2)
  attractiveness <- replace(attractiveness,attractiveness>highEps2,highEps2)
  
  return (list(attractiveness))
  
} # End FRApred

FRApred1 <- function(i, iV, s, j, FRASims, theta) {
  
  wAll <- theta[1]
  wNoth <- theta[2]
  wLef <- theta[3]
  wIn <- theta[4]
  alpha <- theta[5]
  beta <- theta[6]
  gamma <- theta[7]
  delta <- theta[8]
  epsilon <- theta[9]
  zeta <- theta[10]
  
  # First we calculate the prior probabilities
  aux <- c(wAll, wNoth, wLef, wLef, wLef, wLef, wIn, wIn)
  # The probability of region 'RS' is 1 - the sum of the other probabilities
  if (sum(aux) > 1) {
    aux <- aux/sum(aux)
  }
  bias <- c(1 - sum(aux), aux)
  h <- 0 # Shaky hand parameter
  bias[1] <- bias[1] + h
  #  imprimir(bias)
  
  # Start from biases
  attractiveness <- bias
  # Add WinStay
  index <- which(regiones == i)
  # adding win stay only to focal regions
  if (i != 'RS') {
    attractiveness[index] <- attractiveness[index] + alpha * sigmoid(s, beta, gamma) 
  }
  #  print('Attractiveness with WS:')
  #  imprimir(attractiveness)
  
  similarities <- sigmoid(unlist(FRASims), epsilon, zeta)
  similarities <- c(0, unlist(similarities))
  attractiveness <- attractiveness + similarities
  
  attractiveness <- replace(attractiveness,attractiveness<lowerEps2,lowerEps2)
  attractiveness <- replace(attractiveness,attractiveness>highEps2,highEps2)
  
  return (list(attractiveness))
  
} # End FRApred1

FRApred2 <- function(i, iV, s, j, FRASims, theta) {
  
  wAll <- theta[1]
  wNoth <- theta[2]
  wLef <- theta[3]
  wIn <- theta[4]
  alpha <- theta[5]
  beta <- theta[6]
  gamma <- theta[7]
  delta <- theta[8]
  epsilon <- theta[9]
  zeta <- theta[10]
  
  # First we calculate the prior probabilities
  aux <- c(wAll, wNoth, wLef, wLef, wLef, wLef, wIn, wIn)
  # The probability of region 'RS' is 1 - the sum of the other probabilities
  if (sum(aux) > 1) {
    aux <- aux/sum(aux)
  }
  bias <- c(1 - sum(aux), aux)
  #  imprimir(bias)
  
  # Start from 0
  attractiveness <- rep(0, 9)
  # Add WinStay
  index <- which(regiones == i)
  # adding win stay only to focal regions
  if (i != 'RS') {
    attractiveness[index] <- attractiveness[index] + alpha * sigmoid(s, beta, gamma) 
  }
  #  print('Attractiveness with WS:')
  #  imprimir(attractiveness)
  
  similarities <- sigmoid(unlist(FRASims), epsilon, zeta)
  similarities <- c(0, unlist(similarities))
  similarities <- rep(1, 9) + similarities
  similarities <- bias * similarities
  attractiveness <- attractiveness + similarities
  
  attractiveness <- replace(attractiveness,attractiveness<lowerEps2,lowerEps2)
  attractiveness <- replace(attractiveness,attractiveness>highEps2,highEps2)
  
  return (list(attractiveness))
  
} # End FRApred2

# A function to get deviance from WSLS and FRA models
FRAutil <- function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10){
  # Input: theta, parameter vector of length 11
  #        data, the dataframe from which frequencies are obtained
  # Output: Deviance of WSLSpred for all regions and scores
  
  theta <- c(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10)
  
  if (any(is.na(theta))) {
    print('Incorrect parameters: ')
    print(theta)
    return(10000)
  }
  
  # Calculate the probabilities based on FRAWSpred
  #  print('Calculating probabilities')
  args <- args %>%
    dplyr::group_by(RegionFULL, Score, RJcode) %>%
    dplyr::mutate(probs = FRApred1(Region, 
                                   RegionFULL,
                                   Score, 
                                   RJcode,
                                   FRASims,
                                   theta)
    )
  
  # Calculate deviance
  #  print('Calculating deviances')
  args$dev <- mapply(function(x,y) dmultinom(x, prob = y), args$freqs, args$probs)
  
  if (any(is.infinite(args$dev) | is.na(args$dev))) {
    print('Incorrect dev: ')
    #    new_DF <- args[is.infinite(args$dev), ]
    #    print(new_DF)
    #    print(theta)
    #    print(head(args$probs))
    #    print(head(args$freq))
    #    print(head(args$dev))
    return(10000)
  }
  
  a <- -sum(log(args$dev))
  
  if (a == Inf) {
    a <- 10000
  }
  # print(paste("Dev:", a))
  return(a)
  
} # end FRAutil

random_params <- function(x1, x2, x3, x4, x5, x6, x7, x8, x9, x10) {
  
  x1 <- unlist(x1)
  x1 <- c(1, x1)
  #  print(x1)
  a1 <- do.call(runif, as.list(x1))
  
  x2 <- unlist(x2)
  x2 <- c(1, x2)
  #  print(x2)
  a2 <- do.call(runif, as.list(x2))
  
  x3 <- unlist(x3)
  x3 <- c(1, x3)
  #  print(x3)
  a3 <- do.call(runif, as.list(x3))
  
  x4 <- unlist(x4)
  x4 <- c(1, x4)
  #  print(x4)
  a4 <- do.call(runif, as.list(x4))
  
  x5 <- unlist(x5)
  x5 <- c(1, x5)
  #  print(x5)
  a5 <- do.call(runif, as.list(x5))
  
  x6 <- unlist(x6)
  x6 <- c(1, x6)
  #  print(x6)
  a6 <- do.call(runif, as.list(x6))
  
  x7 <- unlist(x7)
  x7 <- c(1, x7)
  #  print(x7)
  a7 <- do.call(runif, as.list(x7))
  
  x8 <- unlist(x8)
  x8 <- c(1, x8)
  #  print(x8)
  a8 <- do.call(runif, as.list(x8))
  
  x9 <- unlist(x9)
  x9 <- c(1, x9)
  #  print(x9)
  a9 <- do.call(runif, as.list(x9))
  
  x10 <- unlist(x10)
  x10 <- c(1, x10)
  #  print(x10)
  a10 <- do.call(runif, as.list(x10))
  
  return(c(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10))
  
} # end random_params

searchFitMLE2 <- function(params, args, max_iter=5) {
  
  contador <- 1
  
  while (contador < max_iter + 1) {
    
    print(paste("Trying mle2...", contador))
    
    fitresFRA <- tryCatch({
      f <- mle2(function(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10) FRAutil(a1, a2, a3, a4, a5, a6, a7, a8, a9, a10),
                start = list(a1=params[1], 
                             a2=params[2], 
                             a3=params[3], 
                             a4=params[4], 
                             a5=params[5], 
                             a6=params[6], 
                             a7=params[7], 
                             a8=params[8], 
                             a9=params[9], 
                             a10=params[10]),
                lower=lower_limits,
                upper=upper_limits,
                method="L-BFGS-B")
      contador <- max_iter + 2
      return(f)
    }, error = function(e) {
      print(paste("Error:", e))
      print(paste("New attempt...(", contador, "/", max_iter, ")", sep=""))
      contador <- contador - 1
      return (NA)
    }
    )
    contador <- contador + 1
  }
  
  return(fitresFRA)  
  
} # end searchFitMLE2

searchFit <- function(params, args, max_iter=10) {
  
  contador <- 1
  
  while (contador < max_iter + 1) {
    
    print(paste("Trying nmkb...", contador))
    
    fitresFRA <- tryCatch({
      f <- nmkb(par=params, 
                fn = function(t) FRAutil(t[1], 
                                         t[2],
                                         t[3],
                                         t[4],
                                         t[5],
                                         t[6], 
                                         t[7], 
                                         t[8], 
                                         t[9], 
                                         t[10]),
                lower=lower_limits,
                upper=upper_limits,
                control=list(trace=0))
      contador <- max_iter + 2
      return(f)
    }, error = function(e) {
      print(paste("Error:", e))
      print(paste("New attempt...(", contador, "/", max_iter, ")", sep=""))
      contador <- contador - 1
      return (NA)
    }
    )
    contador <- contador + 1
  }
  
  return(fitresFRA)  
  
} # end searchFit

searchBestFit <- function(args, N=1, module="nmkb") {
  
  best <- 100000
  fitFRA <- NA
  
  for (n in rep(0, N)) {
    params <- random_params(list(lower_limits[1], upper_limits[1]), 
                            list(lower_limits[2], upper_limits[2]), 
                            list(lower_limits[3], upper_limits[3]), 
                            list(lower_limits[4], upper_limits[4]), 
                            list(lower_limits[5], upper_limits[5]), 
                            list(lower_limits[6], upper_limits[6]), 
                            list(lower_limits[7], upper_limits[7]), 
                            list(lower_limits[8], upper_limits[8]), 
                            list(lower_limits[9], upper_limits[9]), 
                            list(lower_limits[10], upper_limits[10]))
    
    # imprimir(params)
    if (module=="nmkb"){
      bestFit <- searchFit(params, args)
    } else if (module=="mle2") {
      bestFit <- searchFitMLE2(params, args)
    } else {
      print("Invalid module!")
      print("Try with either \'optim\' or \'mle2\'")
      return(NA)
    }
    
    b <- tryCatch({
      bestFit$value
    }, error = function(e){
      print(paste("Oops, optimizer didn\'t work this time!"))
      return(1000000) 
    })
    
    if (b < best) {
      fitFRA <- bestFit
      best <- bestFit$value
    }
    
    print(paste("Best value so far:", best))
  }
  
  b <- tryCatch({
    print(fitFRA$message)
    print(paste("Dev:", fitFRA$value))
    imprimir(fitFRA$par)
    archivo <- paste("Parameter_fit_", module, ".csv", sep = "")
    df <- data.frame(fitFRA)
    # print(head(df))
    write.csv(df, archivo, row.names = FALSE)
    print(paste("Data saved to", archivo))
    return(fitFRA)
  }, error = function(e){
    print("Optimizer", module, "didn\'t work at all :(")
    return(NA) 
  })
  
  return(fitFRA)
  
} # end searchBestFit


ModelProb <- function(regionFrom, regionGo, s, k, theta){
  
  # FRA model returns probability of going from regionFrom to regionGo
  # given FRA similarity to region k
  
  wALL <- theta[1]
  wNOTHING <- theta[2]
  wLEFT <- theta[3]
  wIN <- theta[4]
  alpha <- theta[5]
  beta <- theta[6]
  gamma <- theta[7]
  delta <- theta[8]
  epsilon <- theta[9]
  zeta <- theta[10]
  
  aux <- c(wALL, wNOTHING, wLEFT, wLEFT, wLEFT, wLEFT, wIN, wIN)
  
  # The probability of region 'RS' is 1 - the sum of the other probabilities
  if (sum(aux) > 1) {
    aux <- aux/sum(aux)
  }
  bias <- c(1 - sum(aux), aux)
  #  print('bias')
  #  imprimir(bias)
  
  # Find the attractivenes:
  focal <- 0
  index <- which(regiones == regionGo)
  #  print(paste('Indice regiones:', index))
  attractiveness <- bias[index]
  #  print(paste('attractiveness:', attractiveness))
  
  #  print(paste('Considering case from', regionFrom, 'to', regionGo))
  if (regionFrom == 'RS') { 
    #    print('Entering case from RS')
    if (regionGo != 'RS') {
      #      print('Entering case to Focal')
      if(k == regionGo) {
        attractiveness <- attractiveness + delta * sigmoid(s, epsilon, zeta)
        focal <- delta * sigmoid(s, epsilon, zeta)
      }
    } else {
      #      print('Entering case to RS')
      focal <- delta * sigmoid(s, epsilon, zeta)
    }
  } else {
    #    print('Entering case from Focal')
    if(regionFrom == regionGo) {
      attractiveness <- attractiveness + delta
      focal <- delta
    } else if(k == regionGo) {
      attractiveness <- attractiveness +  delta * sigmoid(s, epsilon, zeta)
      focal <- delta * sigmoid(s, epsilon, zeta)
    }
  }
  
  probab <- attractiveness / (1 + focal)
  
  return(probab)
} # end ModelProb