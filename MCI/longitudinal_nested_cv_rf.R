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

dat <- read.csv('data/mci_progress.csv', stringsAsFactors = T)

dat$X <- NULL
#dat <- smote(last_DX ~ ., dat, perc.over = 7, perc.under = 1)
#summary(dat$last_DX)

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary,# sampling='smote',
                     verboseIter = F)


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
    
    rfRadialGrid <- expand.grid(.mtry = 1:ncol(training))
                                #, maxdepth = 1:10)
    
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    # training <- predict(impute_train, training)
    # 
    # impute_test <- preProcess(rbind(training[,-1], test[,-1]),
    #                           method = "knnImpute")
    
    test[,-1] <- predict(impute_train, test[,-1])
    
    booted_training <- bootstrapping(training)
    
    # tuning
    rfRadialModel <- train(last_DX ~ ., training, method = "rf", 
                           metric = "ROC",
                           preProc = c("knnImpute", "scale", "center"),
                           na.action = na.pass,
                           tuneGrid = rfRadialGrid,
                           trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(rfRadialModel, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(rfRadialModel, test)
    
    # partial ROCs
    # nrow_test <- nrow(test)
    # newPrediction<-c(NA)
    # length(newPrediction) <- nrow_test
    # 
    # for (i in 1:nrow_test){
    # 
    #   rfROC <- roc(evalResults[-i, 'last_DX'], evalResults[-i, 'rf'],
    #              levels = c('CN', 'MCI_AD'))
    #   #rfROC
    #   #plot(rfROC, legacy.axes=T)
    # 
    #   # alternative cutoff
    #   # rfThresh<-coords(rfROC, x='best', best.method='youden')
    #   rfThresh <- coords(rfROC, x = 'best', best.method = 'closest.topleft')
    #   # rfThresh
    # 
    #   # new predictions
    #   newPrediction[i] <- ifelse(evalResults[i, 'rf'] >= rfThresh[1],
    #                             'MCI_AD', 'CN')
    # 
    # }
    
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

write.csv(mcPerf, 'data/mci_rfMcPerf.csv')
write.csv(totalprobabilities, 'data/mci_rf_total_probabilities.csv')

stopCluster(cl)

# metrics <- read.csv('data/rfPerf_100_mc.csv')
# 
# all_metrics_summarised  <- metrics %>%
#   summarise(mean = across(ROC:Kappa, mean),
#             sd = across(ROC:Kappa, sd))
