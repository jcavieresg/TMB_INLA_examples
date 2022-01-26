rm(list = ls())
setwd("C:/Users/Usuario/Desktop/Projects/2022/Germany")

library(TMB)
library(tmbstan)
library(rstan)
library(bayesplot)
library(tictoc)
require(MCMCvis)

compile("linear_bayes.cpp")
dyn.load(dynlib("linear_bayes"))

data = read.table("TeethNitrogen.txt",header = T)
plot(data$Age,data$X15N,  type = "p", col = "blue", ylab = "Conc. Nitrogren", xlab = "Age")
head(data)

#======================================
#                TMB DATA
#======================================
tmb_data = list(y     = as.vector(data$X15N),  # Response
                x     = as.vector(data$Age))   # Covariable


#======================================
#                TMB parameter
#======================================
tmb_par = list(beta0 = 0,
               beta1 = 0,
               logsigma = 0)


#======================================
#                Run the model
#======================================
obj = MakeADFun(data = tmb_data, parameters = tmb_par, DLL = "linear_bayes")
opt = nlminb(obj$par,obj$fn, obj$gr)
rep = sdreport(obj)

plot(data$Age,data$X15N,  type = "p", col = "blue", ylab = "Conc. Nitrogren", xlab = "Age")
abline(a = rep$par.fixed[1], b = rep$par.fixed[2])




#===========================================================================================
#                                        Fit with tmbstan
#===========================================================================================

library(tictoc)
tic("Time of estimation")
fit = tmbstan(obj, chains= 3, open_progress = FALSE, 
              init='last.par.best', control = list(max_treedepth = 10, adapt_delta = 0.9), 
              iter=3000, warmup=500, seed=483892929)
toc()


c_light <- c("#DCBCBC")
c_light_highlight <- c("#C79999")
c_mid <- c("#B97C7C")
c_mid_highlight <- c("#A25050")
c_dark <- c("#8F2727")
c_dark_highlight <- c("#7C0000")

library(bayesplot)
traceplot(fit, pars=names(obj$par), inc_warmup=TRUE)

library(MCMCvis)
MCMCsummary(fit, round = 2)

params_cp <- as.data.frame(fit)
names(params_cp) <- gsub("chain:1.", "", names(params_cp), fixed = TRUE)
names(params_cp) <- gsub("[", ".", names(params_cp), fixed = TRUE)
names(params_cp) <- gsub("]", "", names(params_cp), fixed = TRUE)
params_cp$iter <- 1:7500

# beta0
par(mfrow=c(3,2),mar=c(4,4,0.5,0.5), oma=c(2,3,1,1))
plot(params_cp$iter, params_cp$beta0, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="beta0", cex.lab=1.3, cex.axis=1.3)

running_means_beta0 = sapply(params_cp$iter, function(n) mean(params_cp$beta0[1:n]))
plot(params_cp$iter, running_means_beta0, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of beta0")
abline(h=mean(running_means_beta0), col="grey", lty="dashed", lwd=3)



# beta1
plot(params_cp$iter, params_cp$beta1, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="beta1",  cex.lab=1.3, cex.axis=1.3)

running_means_beta1 = sapply(params_cp$iter, function(n) mean(params_cp$beta1[1:n]))
plot(params_cp$iter, running_means_beta1, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of beta1")
abline(h=mean(running_means_beta1), col="grey", lty="dashed", lwd=3)


# logsigma
plot(params_cp$iter, params_cp$logsigma, col=c_dark, pch=16, cex=0.8, type = "l",
     xlab="Iteration", ylab="logsigma",  cex.lab=1.3, cex.axis=1.3)

running_means_sigma = sapply(params_cp$iter, function(n) mean(params_cp$logsigma[1:n]))
plot(params_cp$iter, running_means_sigma, col=c_dark, pch=16, cex=0.8,  cex.lab=1.3, cex.axis=1.3,
     xlab="Iteration", ylab="MCMC mean of logsigma")
abline(h=mean(running_means_sigma), col="grey", lty="dashed", lwd=3)
mtext("Convergence of the parameters beta0, beta1 and sigma", outer=TRUE,  cex=1, line=-0.5)


# Divergences
divergent = get_sampler_params(fit, inc_warmup=FALSE)[[1]][,'divergent__']
sum(divergent)



