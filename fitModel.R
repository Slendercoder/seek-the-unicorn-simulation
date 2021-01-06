source("MODELpred.R")

###############################################################
# Parameter recovery function
###############################################################

fitModels2Data <- function(args) {
  
  Trials <- 5
  parametros <- list(rep(0, 11), rep(0, 11), rep(0, 11))
  devs <- c(100000, 100000, 100000)
  print("Fitting MBiases...")
  f_MBi <- searchBestFit_MBiases(args, N=Trials, module="nmkb", contador, FALSE)
  print("Fitting WSLS...")
  f_WSLS <- searchBestFit_WSLS(args, N=Trials, module="nmkb", contador, FALSE)
  print("Fitting FRA...")
  f_FRA <- searchBestFit_FRA(args, N=Trials, module="nmkb", contador, FALSE)
  print("--------------")
  tryCatch({
    print(cat("MBiases dev: ",f_MBi$value))
    imprimir(f_MBi$par)
    parametros[[1]] <- c('MBiases', f_MBi$par, rep(0,6))
    devs[1] <- f_MBi$value
  }, error = function(e) {
    print("Optimizer didn't work for MBiases")
  })
  print("--------------")
  tryCatch({
    print(cat("WSLS dev: ",f_WSLS$value))
    imprimir(f_WSLS$par)
    parametros[[2]] <- c('WSLS', f_WSLS$par, rep(0,3))
    devs[2] <- f_WSLS$value
  }, error = function(e) {
    print("Optimizer didn't work for WSLS")
  })
  print("--------------")
  tryCatch({
    print(cat("FRA dev: ",f_FRA$value))
    imprimir(f_FRA$par)
    parametros[[3]] <- c('FRA', f_FRA$par)
    devs[3] <- f_FRA$value
  }, error = function(e) {
    print("Optimizer didn't work for FRA")
  })
  
  data <- as.data.frame(do.call(rbind, parametros))
  names(data) <- c('Model', 'wA', 'wN', 'wL', 'wI',
                   'alpha', 'beta', 'gamma',
                   'delta', 'epsilon', 'zeta')
  data$dev <- devs
  return(data)
  
} # end fitModels2Data

####################################################

archivo <- "./Data/humans_tolerance0.csv"
print(paste("Loading and preparing data", archivo, "..."))
df = read.csv(archivo)

df$Region <- df$Category
df <- find_joint_region(df)
df$RegionFULL <- unlist(df$RegionFULL)
df$RegionGo <- factor(df$RegionGo, levels = regiones)
print(head(df))
args <- getFreqFRA(df, theta)
args <- get_FRASims_list(args)
print(head(args))
print("Data prepared!")

print("Fitting MBiases...")
f_MBiases <- searchBestFit_MBiases(args, N=2, module="nmkb", contador, FALSE)
print(f_MBiases)

print("Fitting WSLS...")
f_WSLS <- searchBestFit_WSLS(args, N=2, module="nmkb", contador, FALSE)
print(f_WSLS)

print("Fitting FRA...")
f_FRA <- searchBestFit_FRA(args, N=2, module="nmkb", contador, FALSE)
print(f_FRA)

fitdata <- fitModels2Data(args)
write.csv(fitdata, './Data/parameter_fit_humans.csv', row.names=FALSE)
