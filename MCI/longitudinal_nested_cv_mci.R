library(caret)
library(tidyverse)
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

cn_progress <- data_numeric[data_numeric$DXMCI == 1,]
cn_progress$last_DX <- factor(ifelse(cn_progress$last_DX == 'Dementia',
                                       'Dementia', 'CN_MCI'),
                          levels = c('CN_MCI', 'Dementia')) 

cn_progress$DXCN <- NULL
cn_progress$DXDementia <- NULL
cn_progress$DXMCI <- NULL
dat <- cn_progress

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
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    gbmModel<-train(last_DX ~ ., training, method = "gbm", 
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
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN_MCI',
                               ifelse(totalnewPrediction == 2,
                                      'Dementia', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('CN_MCI',
                                                              'Dementia'))
  
  # confusion matrix all dataset
  cm <- confusionMatrix(totalnewPrediction, dat$last_DX, positive = 'Dementia')
  cm
  
  # perf
  rfROCfull <- roc(dat$last_DX, totalprobabilities, levels = c('CN_MCI',
                                                               'Dementia'))
  
  v <- c(ROC = auc(rfROCfull), cm$byClass[c(1, 2)], cm$overall[c(1, 2)])
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

post <- Sys.time()

post - pre

write.csv(mcPerf, 'data/mci_gbmMcPerf.csv')
write.csv(totalprobabilities, 'data/mci_gbm_total_probabilities.csv')

stopCluster(cl)

# metrics <- read.csv('data/rfPerf_100_mc.csv')
# 
# all_metrics_summarised  <- metrics %>%
#   summarise(mean = across(ROC:Kappa, mean),
#             sd = across(ROC:Kappa, sd))
