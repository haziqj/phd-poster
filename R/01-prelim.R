## ---- prelim ----
library(iprobit)
library(iprior)
library(ggplot2)
gg_colour_hue <- function(n) {
  hues = seq(15, 375, length = n + 1)
  hcl(h = hues, l = 65, c = 100)[1:n]
}
# library(gganimate)
library(animation)
library(reshape2)
library(directlabels)
library(rstan)
rstan_options(auto_write = TRUE)
options(mc.cores = parallel::detectCores())
stan2coda <- function(fit) {
  mcmc.list(lapply(1:ncol(fit), function(x) mcmc(as.array(fit)[,x,])))
}
library(ggmcmc)
library(coda)

# Function to specify decimal places
decPlac <- function(x, k = 2) format(round(x, k), nsmall = k)

# Function to determine even numbers
isEven <- function(x) x %% 2 == 0

# logit and expit
logit <- function(x) log(x) - log(1 - x)
expit <- function(x, log.expit = FALSE) {
  res <- -log(1 + exp(-x))
  if (isTRUE(log.expit)) res
  else exp(res)
}

# SE kernel
fnH4 <- function(x, y = NULL, l = 1) {
  x <- scale(x, scale = FALSE)
  if (is.vector(x))
    x <- matrix(x, ncol = 1)
  n <- nrow(x)
  A <- matrix(0, n, n)
  index.mat <- upper.tri(A)
  index <- which(index.mat, arr.ind = TRUE)
  xcrossprod <- tcrossprod(x)
  if (is.null(y)) {
    tmp1 <- diag(xcrossprod)[index[, 1]]
    tmp2 <- diag(xcrossprod)[index[, 2]]
    tmp3 <- xcrossprod[index]
    A[index.mat] <- tmp1 + tmp2 - 2 * tmp3
    A <- A + t(A)
    tmp <- exp(-A / (2 * l ^ 2))
  } else {
    if (is.vector(y))
      y <- matrix(y, ncol = 1)
    else y <- as.matrix(y)
    y <- sweep(y, 2, attr(x, "scaled:center"), "-")
    m <- nrow(y)
    B <- matrix(0, m, n)
    indexy <- expand.grid(1:m, 1:n)
    ynorm <- apply(y, 1, function(z) sum(z ^ 2))
    xycrossprod <- tcrossprod(y, x)
    tmp1 <- ynorm[indexy[, 1]]
    tmp2 <- diag(xcrossprod)[indexy[, 2]]
    tmp3 <- as.numeric(xycrossprod)
    B[, ] <- tmp1 + tmp2 - 2 * tmp3
    tmp <- exp(-B / (2 * l ^ 2))
  }
  tmp
}

## ---- points ----
set.seed(123)
N <- 150
f <- function(x, truth = FALSE) {
  35 * dnorm(x, mean = 1, sd = 0.8) +
    65 * dnorm(x, mean = 4, sd = 1.5) +
    (x > 4.5) * (exp((1.25 * (x - 4.5))) - 1) +
    3 * dnorm(x, mean = 2.5, sd = 0.3)
}
x <- c(seq(0.2, 1.9, length = N * 5 / 8), seq(3.7, 4.6, length = N * 3 / 8))
x <- sample(x, size = N)
x <- x + rnorm(N, sd = 0.65)  # adding random fluctuation to the x
x <- sort(x)
y.err <- rt(N, df = 1)
y <- f(x) + sign(y.err) * pmin(abs(y.err), rnorm(N, mean = 4.1))  # adding random terms to the y

# True values
x.true <- seq(-2.1, 7, length = 1000)
y.true <- f(x.true, TRUE)

# Data for plot
dat <- data.frame(x, y)
dat.truth <- data.frame(x.true, y.true)

p1 <- ggplot() +
  geom_point(data = dat, aes(x = x, y = y)) +
  scale_x_continuous(
    limits = c(min(x.true), max(x.true)),
    breaks = NULL, name = expression(italic(x))
  ) +
  scale_y_continuous(
    # limits = c(min(y) - 5, max(y) + 5),
    breaks = NULL, name = expression(italic(y))
  ) +
  coord_cartesian(ylim = c(min(y) - 5, max(y) + 5)) +
  theme_bw()
