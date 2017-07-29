library(MortalitySmooth)
library(iprior)

x <- 1947:2001
x.new <- 2002:2011
e <- selectHMDdata(country = "Denmark", data = "Exposures", sex = "Males",
                   ages = 60, years = x)
e.new <- selectHMDdata(country = "Denmark", data = "Exposures", sex = "Males",
                   ages = 60, years = x.new)
y <- selectHMDdata(country = "Denmark", data = "Deaths", sex = "Males",
                   ages = 60, years = x)
y.new <- selectHMDdata(country = "Denmark", data = "Deaths", sex = "Males",
                       ages = 60, years = x.new)
rates <- selectHMDdata(country = "Denmark", data = "Rates", sex = "Males",
                           ages = 60, years = x)
rates.new <- selectHMDdata(country = "Denmark", data = "Rates", sex = "Males",
                           ages = 60, years = x.new)

mod <- kernL(y, x - 1946, as.numeric(e), model = list(kernel = c("FBM", "FBM")))
mod.fit <- iprior(mod, control = list(maxit = 100000, report = 1000))
mod.fit <- ipriorOptim(mod)
plot(x, y); lines(x, fitted(mod.fit))
plot(c(x, x.new), (c(y, y.new))); lines(x, fitted(mod.fit))
lines(c(x, x.new), predict(mod.fit, list(matrix(c(x, x.new) - 1946),
                                         matrix(c(e, e.new)))), col = 2, lty = 2)

y.hat <- predict(mod.fit, list(matrix(c(x, x.new) - 1946),
                               matrix(c(e, e.new))))
rates.hat <- y.hat / c(e, e.new)
plot(c(x, x.new), log(c(rates, rates.new)))
lines(c(x, x.new), log(rates.hat))

mod.fit2 <- iprior(log(rates), x, e / 1000, model = list(kernel = c("FBM", "FBM")))
# mod.fit2 <- ipriorOptim(mod.fit2$ipriorKernel)
plot(c(x, x.new), log(c(rates, rates.new))); lines(x, fitted(mod.fit2))
lines(c(x, x.new), predict(mod.fit2, list(matrix(c(x, x.new)),
                                          matrix(c(e, e.new)/1000))), col = 2, lty = 2)

mydata <- selectHMDdata(country = country, data = "Rates", sex = sex, ages = age, years = x)
y <- selectHMDdata(country, "Deaths", sex, ages = age, years = x)
e <- selectHMDdata(country, "Exposures", sex, ages = age, years = x)
plot(mydata)

mod <- iprior(log(mydata), x, model = list(kernel = "FBM"))
fit1D <- Mort1Dsmooth(x = x, y = y, offset = log(e))
plot(fit1D); lines(x, fitted(mod))


fit1D <- Mort1Dsmooth(x = x, y = y, offset = log(e))
plot(fit1D)

selectHMDdata("Denmark", "Deaths", "Males", ages = 60, years = 2012)
