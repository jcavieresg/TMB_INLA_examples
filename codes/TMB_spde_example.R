rm(list=ls())
setwd("")

## Libraries used
library(pacman)
pacman::p_load(TMB, TMBhelper, pracma, dplyr, expm, tmbstan, parallel, MASS, Matrix,
               ggplot2, gridExtra, bayesplot, grid, VGAM, stats, scatterplot3d)

options(scipen=999)
# Calculate the number of cores
no_cores <- detectCores() - 1

#============================================
# Load the TMB model
TMB::compile('TMB_spde_example.cpp')
dyn.load( dynlib("TMB_spde_example") )
#============================================

# Data of SPDEtoy
head(SPDEtoy)

mesh = inla.mesh.create(as.matrix(SPDEtoy[, 1:2]),    plot.delay=NULL, refine=FALSE)
mesh$n
plot(mesh)

spde = inla.spde2.matern(mesh, alpha=2)


#================================
# SPDE-based
#================================
data_tmb = list(y_i = as.vector(SPDEtoy$y), 
            site_i = mesh$idx$loc-1,
            M0 = spde$param.inla$M0, 
            M1 = spde$param.inla$M1, 
            M2 = spde$param.inla$M2 )


par_tmb = list(logsigma  = -0.1,  
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
#Opt_spde = TMBhelper::Optimize( obj=Obj, newtonsteps=1, bias.correct=TRUE )
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
fit = tmbstan(obj, chains= 3, open_progress = FALSE, 
              init='last.par.best', control = list(max_treedepth = 10, adapt_delta = 0.9), 
              iter=3000, warmup=700, seed=483892929)
toc()


MCMCsummary(fit, round = 2)

## ESS and Rhat from rstan::monitor
mon = monitor(fit)
max(mon$Rhat)
min(mon$Tail_ESS)

# evalaute problem of convergence
sum(mon$Rhat > 1.01)    
sum(mon$Tail_ESS < 400) 

