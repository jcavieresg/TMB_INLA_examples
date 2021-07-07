#=======================================
#      Spatial modelling with INLA
#=======================================
library(INLA)
library(ggplot2)
library(viridis)
library(cowplot)


# generate some points
set.seed(1234)
n = 1000
coord.x <- runif(1000,0,100)
coord.y <- runif(1000,0,100)
points <- cbind(coord.x, coord.y)

# A random variable with Gaussian distribution
y = rnorm(n,0,1) 

# Plot
df <- data.frame(x = coord.x, y = coord.y, z = y)
ggplot(df, aes(x = x, y = y, col = y)) +
  geom_point() +
  scale_colour_gradient(low="blue", high="green")

# Coordinates
coo = cbind(coord.x, coord.y)

# Grid
mesh = inla.mesh.2d(loc = coo, cutoff = 1, max.edge = c(30, 60))
plot(mesh)
points(coo, col = "blue")

# SPDE method to discretize the spatial domain
spde = inla.spde2.matern(mesh = mesh, alpha = 2, constr = TRUE)

# Spatial index
indexs <- inla.spde.make.index("s", spde$n.spde)

# A matrix
A <- inla.spde.make.A(mesh = mesh, loc = coo)
dim(A)
nrow(coo)
mesh$n

# Borders of the zone
bb <- bbox(coo)
x <- seq(bb[1, "min"] - 1, bb[1, "max"] + 1, length.out = 50)
y <- seq(bb[2, "min"] - 1, bb[2, "max"] + 1, length.out = 50)
coop <- as.matrix(expand.grid(x, y))

ind <- point.in.polygon(coop[, 1], coop[, 2],
                        coo[, 1], coo[, 2])
coop <- coop[which(ind == 1), ]
plot(coop, asp = 1)

Ap <- inla.spde.make.A(mesh = mesh, loc = coop)
dim(Ap)   


# stack for estimation stk.e
stk.est <- inla.stack(tag = "est",
                      data = list(y = df$z),
                      A = list(1, A),
                      effects = list(data.frame(b0 = rep(1, nrow(coo))), s = indexs))

# stack for prediction stk.p
stk.pred <- inla.stack(tag = "pred",
                       data = list(y = NA),
                       A = list(1, Ap),
                       effects = list(data.frame(b0 = rep(1, nrow(coop))), s = indexs))

# stk.full has stk.e and stk.p
stk.full <- inla.stack(stk.est, stk.pred)

#===================================================
#             Fit the model
#===================================================
# Formula
formula <- y ~ 0 + b0 + f(s, model = spde)

# Model
res <- inla(formula,
            data = inla.stack.data(stk.full),
            control.predictor = list(compute = TRUE,
                                     A = inla.stack.A(stk.full)),
            verbose=TRUE)


# Results
index <- inla.stack.index(stk.full, tag = "pred")$data
pred_mean <- res$summary.fitted.values[index, "mean"]
pred_ll <- res$summary.fitted.values[index, "0.025quant"]
pred_ul <- res$summary.fitted.values[index, "0.975quant"]

dpm <- rbind(data.frame(east = coop[, 1], north = coop[, 2],
                        value = pred_mean, variable = "pred_mean"),
             data.frame(east = coop[, 1], north = coop[, 2],
                        value = pred_ll, variable = "pred_ll"),
             data.frame(east = coop[, 1], north = coop[, 2],
                        value = pred_ul, variable = "pred_ul"))
dpm$variable <- as.factor(dpm$variable)


ggplot(dpm) + geom_tile(aes(east, north, fill = value)) +
  facet_wrap(~variable, nrow = 1) +
  coord_fixed(ratio = 1) +
  scale_fill_gradient(
    name = "y variable",
    low = "blue", high = "orange"
  ) +
  theme_bw()

# Spatial random field projection
newloc <- cbind(c(219, 678, 818), c(20, 20, 160))
Aproj <- inla.spde.make.A(mesh, loc = newloc)
Aproj %*% res$summary.random$s$mean

rang <- apply(mesh$loc[, c(1, 2)], 2, range)
proj <- inla.mesh.projector(mesh,
                            xlim = rang[, 1], ylim = rang[, 2],
                            dims = c(300, 300)
)


mean_s <- inla.mesh.project(proj, res$summary.random$s$mean)
sd_s <- inla.mesh.project(proj, res$summary.random$s$sd)


df <- expand.grid(x = proj$x, y = proj$y)
df$mean_s <- as.vector(mean_s)
df$sd_s <- as.vector(sd_s)


mean <- ggplot(df, aes(x = x, y = y, fill = mean_s)) +
  geom_raster() +
  scale_fill_viridis(na.value = "transparent") +
  coord_fixed(ratio = 1) + theme_bw()

sd <- ggplot(df, aes(x = x, y = y, fill = sd_s)) +
  geom_raster() +
  scale_fill_viridis(na.value = "transparent") +
  coord_fixed(ratio = 1) + theme_bw()

plot_grid(mean, sd)