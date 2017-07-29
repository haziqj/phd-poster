source("01-prelim.R")

## ---- plot.function.iprior ----
dev_SEkern_iprior <- function(theta, y = y) {
  alpha <- mean(y)
  lambda <- exp(theta[1])
  psi <- exp(theta[2])
  n <- length(y)
  H <- fnH4(x, l = exp(theta[3]))
  tmp <- eigen(lambda * H)
  u <- tmp$val ^ 2 + 1 / psi
  V <- tmp$vec
  res <- -(n / 2) * log(2 * pi) - (1 / 2) * sum(log(u)) -
    (1 / 2) * ((y - alpha) %*% V) %*% ((t(V) / u) %*% (y - alpha))
  as.numeric(-2 * res)
}

dev_FBMkern_iprior <- function(theta, y = y) {
  alpha <- mean(y)
  lambda <- exp(theta[1])
  psi <- exp(theta[2])
  # theta[3] <- 0
  n <- length(y)
  H <- fnH3(x, gamma = expit(theta[3]))
  tmp <- eigen(lambda * H)
  u <- tmp$val ^ 2 + 1 / psi
  V <- tmp$vec
  res <- -(n / 2) * log(2 * pi) - (1 / 2) * sum(log(u)) -
    (1 / 2) * ((y - alpha) %*% V) %*% ((t(V) / u) %*% (y - alpha))
  as.numeric(-2 * res)
}

dev_Cankern_iprior <- function(theta, y = y) {
  alpha <- mean(y)
  lambda <- exp(theta[1])
  psi <- exp(theta[2])
  n <- length(y)
  H <- fnH2(x)
  tmp <- eigen(lambda * H)
  u <- tmp$val ^ 2 + 1 / psi
  V <- tmp$vec
  res <- -(n / 2) * log(2 * pi) - (1 / 2) * sum(log(u)) -
    (1 / 2) * ((y - alpha) %*% V) %*% ((t(V) / u) %*% (y - alpha))
  as.numeric(-2 * res)
}

plot1_iprior <- function(kernel = "SE", no.of.draws = 100) {
  # Fit an I-prior model -------------------------------------------------------
  if (kernel == "SE") {
    mod <- optim(c(1, 1, 1), dev_SEkern_iprior, method = "L-BFGS", y = y)
    n <- length(y)
    H <- fnH4(x, l = exp(mod$par[3]))
    alpha <- mean(y)
    lambda <- exp(mod$par[1])
    psi <- exp(mod$par[2])
    Vy <- psi * (lambda * H) %*% (lambda * H) + diag(1 / psi, n)
    w.hat <- psi * lambda * H %*% solve(Vy, y - alpha)
    H.star <- fnH4(x = x, y = x.true, l = exp(mod$par[3]))
    y.fitted <- as.numeric(mean(y) + lambda * H.star %*% w.hat)
    y.fitted2 <- as.numeric(mean(y) + lambda * H %*% w.hat)

    # Prior variance for f
    # H.all <- fnH4(x.true, l = exp(mod$par[3]))
    # Vf.pri <- psi * (lambda * H.all) %*% (lambda * H.all)
    Vf.pri <- psi * lambda ^ 2 * tcrossprod(H.star)

    # Posterior variance for f
    Vf.pos <- lambda ^ 2 * H.star %*% solve(Vy, t(H.star))
  } else if (kernel == "FBM") {
    mod <- optim(c(1, 1, 0), dev_FBMkern_iprior, method = "L-BFGS", y = y)
    n <- length(y)
    H <- fnH3(x, gamma = expit(mod$par[3])); class(H) <- NULL
    alpha <- mean(y)
    lambda <- exp(mod$par[1])
    psi <- exp(mod$par[2])
    Vy <- psi * (lambda * H) %*% (lambda * H) + diag(1 / psi, n)
    w.hat <- psi * lambda * H %*% solve(Vy, y - alpha)
    H.star <- fnH3(x = x, y = x.true, gamma = expit(mod$par[3]))
    class(H.star) <- NULL
    y.fitted <- as.numeric(mean(y) + lambda * H.star %*% w.hat)
    y.fitted2 <- as.numeric(mean(y) + lambda * H %*% w.hat)

    # Prior variance for f
    # H.all <- fnH3(x.true, gamma = expit(mod$par[3]))
    # Vf.pri <- psi * (lambda * H.all) %*% (lambda * H.all)
    Vf.pri <- psi * lambda ^ 2 * tcrossprod(H.star)
    class(Vf.pri) <- NULL

    # Posterior variance for f
    Vf.pos <- lambda ^ 2 * H.star %*% solve(Vy, t(H.star))
    class(Vf.pos) <- NULL
  } else if (kernel == "Canonical") {
    mod <- optim(c(1, 1), dev_Cankern_iprior, method = "L-BFGS", y = y)
    n <- length(y)
    H <- fnH2(x); class(H) <- NULL
    alpha <- mean(y)
    lambda <- exp(mod$par[1])
    psi <- exp(mod$par[2])
    Vy <- psi * (lambda * H) %*% (lambda * H) + diag(1 / psi, n)
    w.hat <- psi * lambda * H %*% solve(Vy, y - alpha)
    H.star <- fnH2(x = x, y = x.true); class(H.star) <- NULL
    y.fitted <- as.numeric(mean(y) + lambda * H.star %*% w.hat)
    y.fitted2 <- as.numeric(mean(y) + lambda * H %*% w.hat)

    # Prior variance for f
    Vf.pri <- psi * lambda ^ 2 * tcrossprod(H.star)
    class(Vf.pri) <- NULL

    # Posterior variance for f
    Vf.pos <- lambda ^ 2 * H.star %*% solve(Vy, t(H.star))
    class(Vf.pos) <- NULL
  }

  # Prepare random draws from prior and posterior ------------------------------
  draw.pri <- t(mvtnorm::rmvnorm(no.of.draws, mean = rep(alpha, 1000),
                                 sigma = Vf.pri))
  draw.pos <- t(mvtnorm::rmvnorm(no.of.draws, mean = y.fitted, sigma = Vf.pos))
  dat.f <- rbind(data.frame(x = x.true, y = y.fitted, type = "Posterior"),
                 data.frame(x = x.true, y = mean(y), type = "Prior"))
  melted.pos <- melt(data.frame(f = draw.pos, x = x.true), id.vars = "x")
  melted.pri <- melt(data.frame(f = draw.pri, x = x.true), id.vars = "x")
  melted <- rbind(cbind(melted.pri, type = "Prior"),
                  cbind(melted.pos, type = "Posterior"))

  # Posterior predictive covariance matrix -------------------------------------
  varystar <- abs(diag(Vf.pos)) + 1 / psi
  dat.fit <- data.frame(x.true, y.fitted, sdev = sqrt(varystar),
                        type = "95% credible interval")

  # Prepare random draws for posterior predictive checks -----------------------
  VarY.hat <- (lambda ^ 2) * H %*% solve(Vy, H) + diag(1 / psi, nrow(Vy))
  ppc <- t(mvtnorm::rmvnorm(no.of.draws, mean = y.fitted2, sigma = VarY.hat))
  melted.ppc <- melt(data.frame(x = x, ppc = ppc), id.vars = "x")
  melted.ppc <- cbind(melted.ppc, type = "Posterior predictive check")

  # Random draws from prior and posterior function -----------------------------
  p2.tmp <- ggplot() +
    geom_point(data = dat, aes(x = x, y = y), col = "grey55", alpha = 0.5) +
    scale_x_continuous(
      limits = c(min(x.true), max(x.true)),
      breaks = NULL, name = expression(italic(x))
    ) +
    scale_y_continuous(
      limits = c(min(y) - 5, max(y) + 5),
      breaks = NULL, name = expression(italic(y))
    ) +
    theme_bw()
  p2 <- p2.tmp +
    geom_line(data = melted, aes(x = x, y = value, group = variable),
              col = "steelblue3", size = 0.19, alpha = 0.5) +
    facet_grid(type ~ .) +
    geom_line(data = dat.f, aes(x = x, y = y), size = 1, linetype = 2,
              col = "grey10")
  p2.prior <- p2.tmp +
    geom_line(data = subset(melted, type == "Prior"),
              aes(x = x, y = value, group = variable),
              col = "steelblue3", size = 0.19, alpha = 0.5) +
    facet_grid(type ~ .)
  p2.prior.line <- p2.prior +
    geom_line(data = subset(dat.f, type == "Prior"), aes(x = x, y = y),
              size = 1, linetype = 2, col = "grey10")
  p2.posterior <- p2.tmp +
    geom_line(data = subset(melted, type == "Posterior"),
              aes(x = x, y = value, group = variable),
              col = "steelblue3", size = 0.19, alpha = 0.5) +
    facet_grid(type ~ .)
  p2.posterior.line <- p2.posterior +
    geom_line(data = subset(dat.f, type == "Posterior"), aes(x = x, y = y),
              size = 1, linetype = 2, col = "grey10")

  # Confidence band for predicted values  --------------------------------------
  p3 <- p1 +
    geom_line(data = dat.fit, aes(x = x.true, y = y.fitted), col = "grey50",
              size = 0.9, linetype = 2) +
    geom_ribbon(data = dat.fit, fill = "grey70", alpha = 0.5,
                aes(x = x.true, ymin = y.fitted - 1.96 * sdev,
                    ymax = y.fitted + 1.96 * sdev)) +
    facet_grid(type ~ .)

  p4 <- p2 +
    geom_line(data = dat.truth, aes(x = x.true, y = y.true),
              col = "red", size = 1, alpha = 0.75) +
    theme(legend.position = "none")

  # Posterior predictive checks ------------------------------------------------
  p5 <- ggplot() +
  scale_x_continuous(breaks = NULL, name = expression(italic(y))) +
  scale_y_continuous(breaks = NULL) +
  geom_line(data = melted.ppc,
            aes(x = value, group = variable, col = "yrep", size = "yrep"),
            stat = "density", alpha = 0.5) +
  geom_line(data = dat, aes(x = y, col = "y", size = "y"), stat = "density") +
    theme(legend.position = "bottom") +
  scale_colour_manual(
    name = NULL, labels = c("Observed", "Replications"),
    values = c("grey10", "steelblue3")
  ) +
  scale_size_manual(
    name = NULL, labels = c("Observed", "Replications"),
    values = c(1.1, 0.19)
  ) +
  facet_grid(type ~ .) +
  theme_bw() +
  theme(legend.position = c(0.9, 0.5))

  list(p2 = p2, p2.prior = p2.prior, p2.posterior = p2.posterior,
       p2.prior.line = p2.prior.line,
       p2.posterior.line = p2.posterior.line, p3 = p3, p4 = p4, p5 = p5)
}

## ---- canonical.kernel.iprior ----
plot.can.iprior <- plot1_iprior("Canonical")

## ---- fbm.kernel.iprior ----
plot.fbm.iprior <- plot1_iprior("FBM")

## ---- se.kernel.iprior ----
plot.se.iprior <- plot1_iprior("SE")

## ---- save.plots.for.presentation ----
ggsave("../figure/points.pdf", p1, width = 6.5, height = 6.5 / 2.25)
# ggsave("figure/can-prior.pdf", plot.can$p2.prior.line,
#        width = 6.5, height = 6.5 / 1.5)
# ggsave("figure/can-posterior.pdf", plot.can$p2.posterior.line,
#        width = 6.5, height = 6.5 / 1.5)
ggsave("../figure/fbm-prior.pdf", plot.fbm$p2.prior.line,
       width = 6.5, height = 6.5 / 1.5)
ggsave("../figure/fbm-posterior.pdf", plot.fbm$p2.posterior.line,
       width = 6.5, height = 6.5 / 1.5)
ggsave("../figure/fbm-posterior-truth.pdf", {
  plot.fbm$p2.posterior +
    geom_line(data = dat.truth, aes(x = x.true, y = y.true), size = 1,
              alpha = 0.75, col = "red3") +
    annotate("text", label = "Truth", col = "red3", x = max(x.true),
             y = max(y.true) + 1)
  }, width = 6.5, height = 6.5 / 1.5)
p1 <- p1 +
  geom_line(data = dat.truth, aes(x = x.true, y = y.true), size = 1,
            alpha = 0.75, col = "red3") +
  scale_x_continuous(
    breaks = NULL, name = NULL
  ) +
  scale_y_continuous(
    breaks = NULL, name = NULL
  ) + theme_classic()
ggsave("../figure/plot-line.pdf", p1, width = 4, height = 4 / 2.25)
ggsave("../figure/credible-interval.pdf", plot.fbm$p3, width = 6.5, height = 6.5 / 1.5)
ggsave("../figure/ppc.pdf", plot.fbm$p5, width = 6.5, height = 6.5 / 1.5)
