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

# cn_progress <- data_numeric %>%
#   dplyr::filter(DXCN == 1) %>% 
#   mutate(last_DX = factor(ifelse(last_DX == 'CN', 'CN', 'MCI_AD'),
#                           levels = c('CN', 'MCI_AD'))) %>% 
#   select(-DXCN, -DXDementia, -DXMCI)

cn_progress <- data_numeric %>%
  dplyr::filter(DXMCI == 1) %>% 
  mutate(last_DX = factor(ifelse(last_DX == 'Dementia', 'Dementia', 'CN_MCI'),
                          levels = c('CN_MCI', 'Dementia'))) %>% 
  select(-DXCN, -DXDementia, -DXMCI)

dat <- cn_progress
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
    
    rfRadialGrid <- expand.grid(mtry = 1:ncol(training), maxdepth = 1:10)
    
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    rfRadialModel<-train(last_DX ~ ., training, method = "rfRules", 
                          metric = "ROC",
                          # preProc = c("center", "scale"),
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
