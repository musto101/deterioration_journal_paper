library(tidyverse)
library(caret)

adni_slim <- read.csv('data/adni_slim.csv')

missing.perc <- apply(adni_slim, 2, function(x) sum(is.na(x))) / nrow(adni_slim)

adni_slim <- adni_slim[, which(missing.perc < 0.5)]

dummies <- dummyVars(last_DX ~., data = adni_slim)
data_numeric <- predict(dummies, newdata= adni_slim)
data_numeric <- as.data.frame(data_numeric)
data_numeric <-data.frame(adni_slim$last_DX, data_numeric)

names(data_numeric)[1] <- 'last_DX'

data_numeric$X <- NULL

cn_progress <- data_numeric[data_numeric$DXCN == 1,]
cn_progress$last_DX <- factor(ifelse(cn_progress$last_DX == 'CN',
                                     'CN', 'MCI_AD'),
                              levels = c('CN', 'MCI_AD')) 

cn_progress$DXCN <- NULL
cn_progress$DXDementia <- NULL
cn_progress$DXMCI <- NULL

write.csv(cn_progress, 'data/cn_progress.csv')