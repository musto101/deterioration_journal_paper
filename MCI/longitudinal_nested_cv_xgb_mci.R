library(caret)
library(dplyr)
library(performanceEstimation)
library(doParallel)
library(pROC)

pre <- Sys.time()

adni_slim <- read.csv('data/adni_slim.csv')

adni_slim <- adni_slim[adni_slim$PTMARRY != 'Unknown',]
adni_slim <- adni_slim[adni_slim$last_visit > 0,]

missing.perc <- apply(adni_slim, 2, function(x) sum(is.na(x))) / nrow(adni_slim)

adni_slim <- adni_slim[, which(missing.perc < 0.5)]

dummies <- dummyVars(last_DX ~., data = adni_slim)
data_numeric <- predict(dummies, newdata= adni_slim)
data_numeric <- as.data.frame(data_numeric)
data_numeric <-data.frame(adni_slim$last_DX, data_numeric)

names(data_numeric)[1] <- 'last_DX'

data_numeric$X <- NULL

cn_progress <- data_numeric[data_numeric$DX.MCI == 1,]
cn_progress$last_DX <- factor(ifelse(cn_progress$last_DX == 'Dementia',
                                     'Dementia', 'CN_MCI'),
                              levels = c('CN_MCI', 'Dementia')) 

cn_progress$DX.CN <- NULL
cn_progress$DX.Dementia <- NULL
cn_progress$DX.MCI <- NULL
dat <- cn_progress

### MC initial
mcPerf <- data.frame(ROC = numeric(), Sens = numeric(), Spec = numeric(),
                     Accuracy = numeric(), Kappa = numeric())
mcRep <- 1

# ctrl <- trainControl(method = 'cv', number = 5, classProbs = T, 
#                      summaryFunction = twoClassSummary,# sampling='smote',
#                      verboseIter = F)

rfRadialGrid <- expand.grid(nrounds = 10, max_depth = 5, 
                            eta = seq(0.03,  0.035, 0.0001),
                            gamma = seq(0, 0.0001, 0.00001),
                            colsample_bytree = 0.75,
                            min_child_weight = 21, 
                            subsample = 0.5)

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
    
    # training$last_DX <- factor(training$last_DX)
    # test$last_DX <- factor(test$last_DX)
    # # missing values imputation
    
    impute_train <- preProcess(training, method = "knnImpute")
    training <- predict(impute_train, training)
    
    impute_test <- preProcess(rbind(training[,-1], test[,-1]),
                              method = "knnImpute")
    
    test[,-1] <- predict(impute_test, test[,-1])
    
    
    # training$last_DX <- factor(training$last_DX)
    # test$last_DX <- factor(test$last_DX)
    
    # tuning
    xgbModel <- train(last_DX ~ ., training, method = 'xgbTree',
                      metric = "ROC",
                      # preProc = c("center", "scale"),
                      tuneGrid = rfRadialGrid,
                      family = 'binomial',
                      trControl =  trainControl(
                        method = 'cv',
                        number = 5,
                        verboseIter = TRUE, 
                        returnData = FALSE,
                        returnResamp = "none",                                         
                        classProbs = TRUE,                                           
                        summaryFunction = twoClassSummary
                      ))
    
    
    
    ### post processing cross evaluation
   
    # ROC 
    evalResults <- data.frame(last_DX = test$last_DX)
    evalResults$rf <- predict(xgbModel, test, type = "prob")[, 1]
    evalResults$newPrediction <- predict(xgbModel,test)
    
    
    totalnewPrediction[folds[[n]]] <- evalResults$newPrediction
    totalprobabilities[folds[[n]]] <- evalResults$rf
    

    
  }
  totalnewPrediction <- ifelse(totalnewPrediction == 1, 'CN_MCI',
                               ifelse(totalnewPrediction == 2,
                                      'Dementia', totalnewPrediction))
  totalnewPrediction <- factor(totalnewPrediction, levels = c('Dementia',
                                                              'CN_MCI'))
  
  dat$last_DX = factor(dat$last_DX, levels = c('Dementia','CN_MCI'))
  
  # confusion matrix all dataset
  cm <- confusionMatrix(totalnewPrediction, dat$last_DX, positive = 'Dementia')
  cm
  
  # perf
  rfROCfull <- roc(dat$last_DX, totalprobabilities, levels = c('Dementia',
                                                               'CN_MCI'))
  
  rfROC <- roc(response = cn_progress$last_DX, totalprobabilities,
               levels = c('Dementia', 'CN_MCI'))
  
  rfThresh <- coords(rfROC, x = 'best', best.method = 'youden', transpose = F)
  pred <- ifelse(totalprobabilities >= rfThresh[1, 1], 'CN_MCI', 'Dementia')
  sen <- sensitivity(factor(pred), (cn_progress$last_DX))
  speci <- specificity(factor(pred), (cn_progress$last_DX))
  kp <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][2]
  acc <- confusionMatrix(factor(pred), (cn_progress$last_DX))[[3]][1]
  
  v <- c(ROC = auc(rfROCfull), sen, speci, acc, kp)
  names(v) <- c('ROC', 'Sens', 'Spec', 'Accuracy', 'Kappa')
  v <- data.frame(t(v))
  
  mcPerf <- rbind(mcPerf, v)
}

#write.csv(mcPerf, 'data/mci_xgbMcPerf.csv')

stopCluster(cl)

post <- Sys.time()

# metrics <- read.csv('data/rfPerf_100_mc.csv')
# 
# all_metrics_summarised  <- metrics %>%
#   summarise(mean = across(ROC:Kappa, mean),
#             sd = across(ROC:Kappa, sd))
