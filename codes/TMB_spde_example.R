rm(list=ls())

setwd("C:/Users/Usuario/Desktop/Lecturas/Tutoriales/INLA_TMB")

## Libraries used
library(TMB)
library(INLA)
library(tmbstan)
library(rstan)
library(bayesplot)
library(tictoc)
library(ggplot2)
library(geoR)
require(MCMCvis)

library(parallel)
# Calculate the number of cores
no_cores <- detectCores() - 1

#============================================
# Load the TMB model
TMB::compile('TMB_spde_example.cpp')
dyn.load( dynlib("TMB_spde_example") )
#============================================

# Data of SPDEtoy
head(SPDEtoy)

mesh = inla.mesh.create(as.matrix(SPDEtoy[1:100, 1:2]), plot.delay=NULL, refine=FALSE)
plot(mesh)

spde = inla.spde2.matern(mesh, alpha=2)

######## ######## SPDE-based
# Build object
Data = list(y_i = as.vector(SPDEtoy$y[1:100]), 
            site_i = mesh$idx$loc-1,
            M0 = spde$param.inla$M0, 
            M1 = spde$param.inla$M1, 
            M2 = spde$param.inla$M2 )

Params = list(logsigma  = -0.1,  
              logtau    = spde$param.inla$theta.initial[1],
              logkappa  = spde$param.inla$theta.initial[2],
              omega_s = rnorm(nrow(spde$param.inla$M0)))

Obj = MakeADFun(data = Data, parameters = Params, random="omega_s", DLL="TMB_spde_example" )


# Optimize
library(tictoc)
tic("Time of estimation")
opt = with(Obj, nlminb(par, fn, gr, control=list(trace=1)))
toc()

AIC = 2*opt$objective +2*length(opt$par)
AIC

# Optimize
#Opt_spde = TMBhelper::Optimize( obj=Obj, newtonsteps=1, bias.correct=TRUE )
h_spde = Obj$env$spHess(random=TRUE)
report_spde = Obj$report()
# Sparseness
image( h_spde )


#===========================================================================================
#                                        Fit with tmbstan
#===========================================================================================

tic("Time of estimation")
fit = tmbstan(Obj, chains= 2, open_progress = FALSE, 
              init='last.par.best', control = list(max_treedepth = 10, adapt_delta = 0.9), 
              iter=2000, warmup=500, seed=483892929)
toc()


c_light <- c("#DCBCBC")
c_light_highlight <- c("#C79999")
c_mid <- c("#B97C7C")
c_mid_highlight <- c("#A25050")
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")


traceplot(fit, pars=names(Obj$par), inc_warmup=TRUE)

MCMCsummary(fit, round = 2)

## ESS and Rhat from rstan::monitor
mon = monitor(fit)
max(mon$Rhat)
min(mon$Tail_ESS)

# evalaute problem of convergence
sum(mon$Rhat > 1.01)    
sum(mon$Tail_ESS < 400) 

params_cp <- as.data.frame(fit)
names(params_cp) <- gsub("chain:1.", "", names(params_cp), fixed = TRUE)
names(params_cp) <- gsub("[", ".", names(params_cp), fixed = TRUE)
names(params_cp) <- gsub("]", "", names(params_cp), fixed = TRUE)
params_cp$iter <- 1:3000

# logtau
par(mfrow=c(3,2),mar=c(4,4,0.5,0.5), oma=c(2,3,1,1))
plot(params_cp$iter, params_cp$logtau, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="logtau", cex.lab=1.3, cex.axis=1.3)

running_means_tau = sapply(params_cp$iter, function(n) mean(params_cp$logtau[1:n]))
plot(params_cp$iter, running_means_tau, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of logtau")
abline(h=mean(running_means_tau), col="grey", lty="dashed", lwd=3)



# logkappa 
plot(params_cp$iter, params_cp$logkappa, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="logkappa",  cex.lab=1.3, cex.axis=1.3)

running_means_kappa = sapply(params_cp$iter, function(n) mean(params_cp$logkappa[1:n]))
plot(params_cp$iter, running_means_kappa, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of logkappa")
abline(h=mean(running_means_kappa), col="grey", lty="dashed", lwd=3)


# logsigma
plot(params_cp$iter, params_cp$logsigma, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="logsigma",  cex.lab=1.3, cex.axis=1.3)

running_means_sigma = sapply(params_cp$iter, function(n) mean(params_cp$logsigma[1:n]))
plot(params_cp$iter, running_means_sigma, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of logsigma")
abline(h=mean(running_means_sigma), col="grey", lty="dashed", lwd=3)
mtext("Convergence of the parameters tau, kappa and sigma", outer=TRUE,  cex=1, line=-0.5)


# Divergences
divergent = get_sampler_params(fit, inc_warmup=FALSE)[[1]][,'divergent__']
sum(divergent)

## Methods provided by 'rstan'
class(fit)
methods(class ="stanfit")

## ESS and Rhat from rstan::monitor
mon = monitor(fit)
max(mon$Rhat)
min(mon$Tail_ESS)

# evalaute problem of convergence
sum(mon$Rhat > 1.01)    
sum(mon$Tail_ESS < 400) 
