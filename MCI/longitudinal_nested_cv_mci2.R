library(caret)
library(tidyverse)
library(performanceEstimation)
library(doParallel)
library(pROC)
pre <- Sys.time()

adni_slim <- read.csv('data/adni_slim.csv')

missing.perc <- apply(adni_slim, 2, function(x) sum(is.na(x))) / nrow(adni_slim)

adni_slim <- adni_slim[, which(missing.perc < 0.5)]
adni_slim <- adni_slim[adni_slim$PTMARRY != 'Unknown',]
adni_slim <- adni_slim[adni_slim$last_visit > 0,]

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

table(cn_progress$last_DX)

cn_progress$last_DX <- as.numeric(cn_progress$last_DX)

table(cn_progress$last_DX)

cn_progress$last_DX <- ifelse(cn_progress$last_DX == 1, 0, 1)

table(cn_progress$last_DX)

#MCI_CN is 0

write.csv(cn_progress, 'data/mci_progress.csv')

dat <- cn_progress

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
                     summaryFunction = twoClassSummary, sampling = 'smote',
                     verboseIter = F)

glmGrid <- expand.grid(lambda = seq(0, 5, 0.1),
                       alpha = seq(0, 1, 0.1))

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
    
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    # tuning
    glmModel <- train(last_DX ~ ., training, method = "glmnet", 
                          metric = "ROC",
                          # preProc = c("center", "scale"),
                          tuneGrid = glmGrid,
                          trControl = ctrl)
    
    
    
    ### post processing cross evaluation
    
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(glmModel, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(glmModel, test)
    
    
    totalnewPrediction[folds[[n]]] <- evalResults$newPrediction
    totalprobabilities[folds[[n]]] <- evalResults$rf
  }
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN_MCI',
                               ifelse(totalnewPrediction == 2,
                                      'Dementia', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('CN_MCI',
                                                              'Dementia'))
  dat$last_DX = factor(dat$last_DX, levels = c('CN_MCI',
                                               'Dementia'))
  
  # confusion matrix all dataset
  cm <- confusionMatrix(totalnewPrediction, dat$last_DX, positive = 'Dementia')
  cm
  
  # perf
  rfROCfull <- roc(dat$last_DX, totalprobabilities, levels = c('CN_MCI',
                                                               'Dementia'))
  rfROC <- roc(response = cn_progress$last_DX, totalprobabilities,
               levels = c('CN_MCI','Dementia'))
  
  rfThresh <- coords(rfROC, x = 'best', best.method = 'youden')
  
  pred <- ifelse(totalprobabilities >= rfThresh[1, 1], 'CN_MCI','Dementia')
  
  sen <- sensitivity(factor(pred), (cn_progress$last_DX))
  speci <- specificity(factor(pred), (cn_progress$last_DX))
  kp <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][2]
  acc <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][1]
  
  v <- c(ROC = auc(rfROCfull), sen, speci, acc, kp)
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

#write.csv(mcPerf, 'data/mci_glm_100_McPerf.csv')
#write.csv(totalprobabilities, 'data/mci_glm_McPerf_probabilities.csv')

stopCluster(cl)

post <- Sys.time()

time_taken <- post - pre

# metrics <- read.csv('data/mci_glmMcPerf_100.csv')
# 
# all_metrics_summarised  <- metrics %>%
#   summarise(mean = across(ROC:Kappa, mean),
#             sd = across(ROC:Kappa, sd))
