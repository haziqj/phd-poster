source("01-prelim.R")

dat <- gen_circle(n = 750, m = 3, sd = 0.09)  # generate 3-class toy example data set
plot(dat)
(mod <- iprobit(y ~ X1 + X2, dat, kernel = "FBM", control = list(maxit = 1500)))
p <- iplot_predict(mod)

# dat.true <- gen_circle(n = 600, m = 3, sd = 0)
# ggplot(as.data.frame(dat.true)) + geom_path(aes(x = X1, y = X2, group = y, col = y))
# p + geom_path(data = as.data.frame(dat.true), aes(x = X1, y = X2, group = y, col = y))

dat.new <- gen_circle(n = 15, m = 3, sd = 0.09)
dat.new.pred <- predict(mod, dat.new)
dat.new.pred$prob[, dat.new$y]
prob.lab <- diag(apply(dat.new.pred$prob, 1, function(x) decPlac(x[dat.new$y], 2)))
p1 <- p +
  geom_label(data = as.data.frame(dat.new),
               aes(x = X1, y = X2, col = dat.new$y, label = prob.lab),
               size = 3.3, show.legend = FALSE, label.size = 0.9) +
  scale_x_continuous(breaks = NULL, expression(italic(X[1]))) +
  scale_y_continuous(breaks = NULL, name = expression(italic(X[2]))) +
  theme(legend.justification = c(1, 0), legend.position = c(1 - 0.015, 0 + 0.02),
        legend.text = element_text(size = 10),
        legend.background = element_rect(size = 0.5, colour = "grey10")) +
  guides(colour = guide_legend(override.aes = list(size = 2.5))); p1
ggsave("../figure/iprobit_multiclass.pdf", p1, width = 3.5 * 18 / 11, height = 3.5)
