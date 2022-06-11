library(caret)
library(tidyverse)
library(performanceEstimation)
library(doParallel)
library(pROC)
pre <- Sys.time()

bootstrapping <- function(training) {
  
  training <- as_tibble(training)
  
  x <- training %>% sample_n(replace = T, size = 1000)
  
  return(x)
}

dat <- read.csv('data/cn_progress.csv', stringsAsFactors = T)

dat$X <- NULL

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary,# sampling='smote',
                     verboseIter = F)

GBMGrid <- expand.grid(n.trees = seq(1, 200, 1), interaction.depth = 1:10,
                       shrinkage = seq(0.1, 0.9, 0.1), n.minobsinnode = 5:10)

cl <- makeCluster(detectCores())
# cl <- makeCluster(7)
registerDoParallel(cl)

#repeat mcRep times

pre <- Sys.time()
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
    
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    # impute_test <- preProcess(rbind(training[,-1], test[,-1]),
    #                           method = "knnImpute")
    
    test[,-1] <- predict(impute_train, test[,-1])
    
    booted_training <- bootstrapping(training)
    
    # tuning
    gbmModel<-train(last_DX ~ ., booted_training, method = "gbm", 
                    metric = "ROC",
                    # preProc = c("center", "scale"),
                    tuneGrid = GBMGrid,
                    trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(gbmModel, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(gbmModel, test)
    
    
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
  
  v <- c(ROC = auc(rfROCfull), cm$byClass[c(1, 2)], cm$overall[c(1, 2)])
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

post <- Sys.time()

post - pre

write.csv(mcPerf, 'data/cn_gbmMcPerf.csv')
write.csv(totalprobabilities, 'data/cn_gbm_total_probabilities.csv')

stopCluster(cl)

#metrics <- read.csv('data/cn_gbmMcPerf.csv')
post - pre
# 
# all_metrics_summarised  <- metrics %>%
#   summarise(mean = across(ROC:Kappa, mean),
#             sd = across(ROC:Kappa, sd))
