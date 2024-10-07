rm(list=ls())
setwd("")

## Libraries used
library(pacman)
pacman::p_load(TMB, TMBhelper, INLA, pracma, dplyr, expm, tmbstan, parallel, MASS, Matrix,
               ggplot2, gridExtra, bayesplot, grid, VGAM, stats, scatterplot3d)

options(scipen=999)
# Calculate the number of cores
no_cores <- detectCores() - 1

#============================================
# Load the TMB model
TMB::compile('TMB_spde_example.cpp')
dyn.load( dynlib("TMB_spde_example") )
#============================================

# Data of SPDEtoy (from INLA package)
head(SPDEtoy)
coords <- as.matrix(SPDEtoy[1:50, 1:2])
mesh = inla.mesh.create(coords, plot.delay=NULL, refine=FALSE)
mesh$n
plot(mesh)

# Create the observation matrix
A <- inla.spde.make.A(mesh = mesh, loc = coords)

# Create the spde model object
spde = inla.spde2.matern(mesh, alpha = 2)
spde_mat = spde$param.inla[c("M0","M1","M2")]

#================================
# Data inputs for TMB
#================================
data_tmb = list(y = as.vector(SPDEtoy$y[1:50]), # first 100 observations
                spde_mat = spde_mat,
                A = A)

#================================
# Pars inputs for TMB
#================================
par_tmb = list(logsigma_e  = 0.1,  
               logtau    = spde$param.inla$theta.initial[1],
               logkappa  = spde$param.inla$theta.initial[2],
               u = rnorm(nrow(spde$param.inla$M0)))

obj = MakeADFun(data = data_tmb, parameters = par_tmb, random="u", DLL="TMB_spde_example" )


# Optimize
library(tictoc)
tic("Time of estimation")
opt = with(obj, nlminb(par, fn, gr, control=list(trace=1)))
toc()

# Optimize
h_spde = obj$env$spHess(random=TRUE)
report_spde = obj$report()
# Sparseness
image( h_spde )

library(scatterplot3d)
scatterplot3d(SPDEtoy$s1, SPDEtoy$s2, obj$report()$preds)


#===========================================================================================
# Bayesian modelling ---> tmbstan
#===========================================================================================
tic("Time of estimation")
fit = tmbstan(obj,
              chains= 3, open_progress = FALSE,
              control = list(max_treedepth= 10,  adapt_delta = 0.8),
              iter = 3000, warmup= 700, cores = no_cores,
              init = 'last.par.best', seed = 12345)
toc()


MCMCsummary(fit, round = 2)

## ESS and Rhat from rstan::monitor
mon = monitor(fit)
max(mon$Rhat)
min(mon$Tail_ESS)

# evalaute problem of convergence
sum(mon$Rhat > 1.01)    
sum(mon$Tail_ESS < 400) 

