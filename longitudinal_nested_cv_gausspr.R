library(caret)
library(dplyr)
library(performanceEstimation)
library(doParallel)
library(pROC)

adni_slim <- read.csv('data/adni_slim.csv')

missing.perc <- apply(adni_slim, 2, function(x) sum(is.na(x))) / nrow(adni_slim)

adni_slim <- adni_slim[, which(missing.perc < 0.5)]

dummies <- dummyVars(last_DX ~., data = adni_slim)
data_numeric <- predict(dummies, newdata= adni_slim)
data_numeric <- as.data.frame(data_numeric)
data_numeric <-data.frame(adni_slim$last_DX, data_numeric)

names(data_numeric)[1] <- 'last_DX'

data_numeric$X <- NULL

cn_progress <- data_numeric %>%
  dplyr::filter(DXCN == 1) %>%
  mutate(last_DX = factor(ifelse(last_DX == 'CN', 'CN', 'MCI_AD'),
                          levels = c('CN', 'MCI_AD'))) %>%
  select(-DXCN, -DXDementia, -DXMCI)


dat <- cn_progress
#summary(dat$last_DX)

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary, verboseIter = F)

rfRadialGrid <- expand.grid(.sigma = seq(1, 2, 0.001))

cl <- makeCluster(detectCores())
# cl <- makeCluster(7)
registerDoParallel(cl)

#repeat mcRep times
for (j in 1:mcRep) {
  # create nrfolds folds and start outer CV
  print(j)
  nrfolds = nrow(dat)/3 
  
  folds <- createFolds(dat$last_DX, k = nrfolds) 
  
  totalnewPrediction <- c(NA)
  length(totalnewPrediction) <- nrow(dat)
  
  totalprobabilities <- c(NA)
  length(totalprobabilities) <- nrow(dat)
  
  for (n in 1:nrfolds){
    
    training <- dat[-folds[[n]],]
    test <- dat[folds[[n]],]
    
    # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    gaussModel <- train(last_DX ~ ., training, method = 'gaussprRadial', 
                      metric = "ROC",
                      # preProc = c("center", "scale"),
                      tuneGrid = rfRadialGrid,
                      trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(gaussModel, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(gaussModel, test)
    
    totalnewPrediction[folds[[n]]] <- evalResults$newPrediction
    totalprobabilities[folds[[n]]] <- evalResults$rf
  }
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN',
                               ifelse(totalnewPrediction == 2,
                                      'MCI_AD', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('CN',
                                                              'MCI_AD'))
  
  # confusion matrix all dataset
  cm <- confusionMatrix(totalnewPrediction, dat$last_DX, positive = 'MCI_AD')
  cm
  
  # perf
  rfROCfull <- roc(dat$last_DX, totalprobabilities, levels = c('CN',
                                                               'MCI_AD'))
  rfROC <- roc(response = cn_progress$last_DX, totalprobabilities,
               levels = c('CN', 'MCI_AD'))
  
  rfThresh <- coords(rfROC, x = 'best', best.method = 'youden')
  
  pred <- ifelse(totalprobabilities >= rfThresh[1, 1], 'MCI_AD', 'CN')
  
  sen <- sensitivity(factor(pred), (cn_progress$last_DX))
  speci <- specificity(factor(pred), (cn_progress$last_DX))
  kp <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][2]
  acc <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][1]
  
  v <- c(ROC = auc(rfROCfull), sen, speci, acc, kp)
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

# write.csv(mcPerf, 'data/cn_gaussMcPerf.csv')
# write.csv(totalprobabilities, 'data/cn_gauss_probabilities.csv')

stopCluster(cl)
