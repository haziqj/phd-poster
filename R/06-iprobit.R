source("01-prelim.R")

dat <- gen_circle(n = 750, m = 3, sd = 0.08)  # generate 3-class toy example data set
plot(dat)
(mod <- iprobit(y ~ X1 + X2, dat, kernel = "FBM", control = list(maxit = 500)))

dat.true <- gen_circle(n = 600, m = 3, sd = 0)
ggplot(as.data.frame(dat.true)) + geom_path(aes(x = X1, y = X2, group = y, col = y))

p <- iplot_predict(mod)
p + geom_path(data = as.data.frame(dat.true), aes(x = X1, y = X2, group = y, col = y))

dat.new <- gen_circle(n = 15, m = 3, sd = 0.08)
dat.new.pred <- predict(mod, dat.new)
p + geom_label(data = as.data.frame(dat.new), aes(x = X1, y = X2, col = dat.new.pred$y,
                                                  label = apply(dat.new.pred$prob, 1, max)))
