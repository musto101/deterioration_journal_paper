library(caret)
library(broom)
library(tidyverse)
library(performanceEstimation)
library(doParallel)
library(pROC)
#library(boot)
pre <- Sys.time()

bootstrapping <- function(training) {
  
  training <- as.tibble(training)
  
  x <- training %>% sample_n(replace = T, size = 1000)
  
  return(x)
}

dat <- read.csv('data/mci_progress.csv', stringsAsFactors = T)

dat$X <- NULL
#dat$last_DX <- as.numeric(dat$last_DX)

#table(dat$last_DX)

#dat <-as.data.frame(lapply(dat,as.numeric))

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 100

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
    #training <- predict(impute_train, training)
    
    # impute_test <- preProcess(rbind(training[,-1], test[,-1]),
    #                           method = "knnImpute")
    
    test[,-1] <- predict(impute_train, test[,-1])
    
    
    booted_training <- bootstrapping(training)
    # tuning
    glmModel <- train(last_DX ~ .,
                      data = booted_training, method = "glmnet",
                      metric = "ROC",
                      na.action = na.pass,
                      preProcess = c("knnImpute", "scale", "center"),
                      tuneGrid = glmGrid, trControl = ctrl)
    
    
    
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
  rfROC <- roc(response = dat$last_DX, totalprobabilities,
               levels = c('CN_MCI','Dementia'))
  
  rfThresh <- coords(rfROC, x = 'best', best.method = 'youden')
  
  pred <- ifelse(totalprobabilities >= rfThresh[1, 1], 'CN_MCI','Dementia')
  
  sen <- sensitivity(factor(pred), (dat$last_DX))
  speci <- specificity(factor(pred), (dat$last_DX))
  kp <- confusionMatrix(factor(pred), (dat$last_DX))[[3]][2]
  acc <- confusionMatrix(factor(pred), (dat$last_DX))[[3]][1]
  
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
