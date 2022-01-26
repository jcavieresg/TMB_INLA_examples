rm(list = ls())
setwd("C:/Users/Usuario/Desktop/Projects/2022/Germany")

library(TMB)


compile("linear_regression.cpp")
dyn.load(dynlib("linear_regression"))

TN = read.table("TeethNitrogen.txt",header = T)

plot(data$x,data$NL)

# data section
data = list()
data$y = TN$X15N[TN$Tooth=="Moby"]
data$x = TN$Age[TN$Tooth=="Moby"]

# parameter section
parameters = list(beta0 = 0.5, beta1 = 0.2, logsigma = 0.3)

# TBM model
obj = MakeADFun(data,parameters,DLL = "linear_regression")
opt = nlminb(obj$par,obj$fn, obj$gr)
rep = sdreport(obj)

plot(data$x,data$y)
abline(a = rep$par.fixed[1], b = rep$par.fixed[2])


rep$cov
rep$value
rep

obj$fn()
