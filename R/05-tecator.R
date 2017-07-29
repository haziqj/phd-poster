data(tecator, package = "caret")
endpoints <- as.data.frame(endpoints)
colnames(endpoints) <- c("water", "fat", "protein")

# Plot of 10 random spectra predicting fat content
# (reproduced from package caret, v6.0-68)
set.seed(1)
inSubset <- sample(1:dim(endpoints)[1], 10)
absorpSubset <- absorp[inSubset,]
endpointSubset <- endpoints$fat[inSubset]
newOrder <- order(absorpSubset[,1])
absorpSubset <- absorpSubset[newOrder,]
endpointSubset <- endpointSubset[newOrder]
plotColors <- rainbow(10)
plot(absorpSubset[1,], type = "n", ylim = range(absorpSubset), xlim = c(0, 105),
     xlab = "Wavelength index", ylab = "Absorption")
for (i in 1:10) {
  points(absorpSubset[i, ], type = "l", col = plotColors[i], lwd = 2)
  text(105, absorpSubset[i, 100], endpointSubset[i], col = plotColors[i])
}
title("Fat content predictor profiles for 10 random samples")

# Prepare data for iprior
# n = 215, use first 160 for training
absorpTrain <- -t(diff(t(absorp)))	# this takes first differences using diff()
absorpTest <- absorpTrain[161:215,]
absorpTrain <- absorpTrain[1:160,]

# Other variables
fatTrain <- endpoints$fat[1:160]
fatTest <- endpoints$fat[161:215]
waterTrain <- endpoints$water[1:160]
waterTest <- endpoints$water[161:215]

# Model 1: Canonical RKHS (linear)
(mod1 <- kernL(y = fatTrain, absorpTrain))
# Model 2: Canonical RKHS (quadratic)
mod2 <- kernL(y = fatTrain, absorpTrain, absorpTrain ^ 2,
              model = list(order = c("1", "1^2")))
# Model 3: Canonical RKHS (cubic)
mod3 <- kernL(y = fatTrain, absorpTrain, absorpTrain ^ 2, absorpTrain ^ 3,
              model = list(order = c("1", "1^2", "1^3")))
# Model 4: FBM RKHS (Hurst = 0.5 by default)
mod4 <- kernL(y = fatTrain, absorpTrain, model = list(kernel = "FBM"))
# Model 5: FBM RKHS + extra covariate
(mod5 <- kernL(y = fatTrain, absorpTrain, waterTrain,
               model = list(kernel = c("FBM", "Canonical"))))

# Fit all models
mod1.fit <- ipriorOptim(mod1, control = list(silent = TRUE))  # linear
mod2.fit <- ipriorOptim(mod2, control = list(silent = TRUE))  # quadratic
mod3.fit <- ipriorOptim(mod3, control = list(silent = TRUE))  # cubic
mod4.fit <- ipriorOptim(mod4, control = list(silent = TRUE))  # smooth
mod5.fit <- fbmOptim(mod4, silent = TRUE)  # smooth, MLE
mod6.fit <- fbmOptim(mod5, silent = TRUE)  # smooth, MLE with extra covariate

fatTestPredicted <- predict(mod1.fit, list(absorpTest))
head(fatTestPredicted)

RMSE.Train1 <- mod1.fit$sigma
fatTestPredicted1 <- predict(mod1.fit, list(absorpTest))
RMSE.Test1 <- sqrt(mean((fatTestPredicted1 - fatTest) ^ 2))

RMSE.Train2 <- mod2.fit$sigma
fatTestPredicted2 <- predict(mod2.fit, list(absorpTest, absorpTest ^ 2))
RMSE.Test2 <- sqrt(mean((fatTestPredicted2 - fatTest) ^ 2))

RMSE.Train3 <- mod3.fit$sigma
fatTestPredicted3 <- predict(mod3.fit,
                             list(absorpTest, absorpTest ^ 2, absorpTest ^ 3))
RMSE.Test3 <- sqrt(mean((fatTestPredicted3 - fatTest) ^ 2))

RMSE.Train4 <- mod4.fit$sigma
fatTestPredicted4 <- predict(mod4.fit, list(absorpTest))
RMSE.Test4 <- sqrt(mean((fatTestPredicted4 - fatTest) ^ 2))

RMSE.Train5 <- mod5.fit$sigma
fatTestPredicted5 <- predict(mod5.fit, list(absorpTest))
RMSE.Test5 <- sqrt(mean((fatTestPredicted5 - fatTest) ^ 2))

RMSE.Train6 <- mod6.fit$sigma
fatTestPredicted6 <- predict(mod6.fit, list(absorpTest, waterTest))
RMSE.Test6 <- sqrt(mean((fatTestPredicted6 - fatTest) ^ 2))

tab <- c(mod1.fit$log.lik, RMSE.Train1, RMSE.Test1)
tab <- rbind(tab, c(mod2.fit$log.lik, RMSE.Train2, RMSE.Test2))
tab <- rbind(tab, c(mod3.fit$log.lik, RMSE.Train3, RMSE.Test3))
tab <- rbind(tab, c(mod4.fit$log.lik, RMSE.Train4, RMSE.Test4))
tab <- rbind(tab, c(mod5.fit$log.lik, RMSE.Train5, RMSE.Test5))
tab <- rbind(tab, c(mod6.fit$log.lik, RMSE.Train6, RMSE.Test6))
rownames(tab) <- c("Linear", "Quadratic", "Cubic", "FBM gam=0.50",
                   paste0("FBM gam=", mod5.fit$ipriorKernel$model$Hurst),
                   paste0("FBM gam=", mod6.fit$ipriorKernel$model$Hurst[1],
                          " + extra cov."))
colnames(tab) <- c("Log-lik", "Training RMSE", "Test RMSE")
tab
