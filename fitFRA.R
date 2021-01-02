#source("Model_Plots.R")
source("FRApred.R")
library(beepr)

###############################################################
# Loading and preparing data...
###############################################################

#archivo <- "../Python Codes/Simulations/M5_full.csv"
#archivo <- "../Python Codes/Simulations/N1_full.csv"
#archivo <- "../Python Codes/Dyads/output-435-261.csv"
#archivo <- "N1_full.csv"
archivo <- "humans_only_absent.csv"
#archivo <- "output-435-261.csv"
print(paste("Loading and preparing data", archivo, "..."))
df = read.csv(archivo)
df <- find_joint_region(df)
df$RegionFULL <- unlist(df$RegionFULL)
df$RegionGo <- factor(df$RegionGo, levels = regiones)
print(head(df))
args <- getFreqFRA(df, theta)
args <- get_FRASims_list(args)
print(head(args))
beep()

###############################################################
# Parameter recovery...
###############################################################

#f <- searchBestFit(args, N=5, module="mle2")
f <- searchBestFit(args, N=1, module="nmkb")
beep()
print(f)

###############################################################
# Plotting Parameter fit
###############################################################

df <- find_joint_region(df)
df$RegionFULL <- unlist(df$RegionFULL)
df$RegionGo <- factor(df$RegionGo, levels = regiones)
df <- get_FRASims(df) # Requires to run df <- find_joint_region(df)
df$RegionFULL <- unlist(df$RegionFULL)
df$RegionGo <- factor(df$RegionGo, levels = regiones)
theta <- f$par
lista_regs <- list(c('DOWN', 'UP'), c('LEFT', 'RIGHT'), c('IN', 'OUT'))
for (regs in lista_regs) {
  #  print(regs)
  p <- plot_FRA_regs(df, regs)
  r <- paste(regs, collapse="-")
  grafico <- paste('PlotFRA-', r, '.pdf',sep="")
  ggsave(grafico, p)
}
