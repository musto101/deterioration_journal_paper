library(tidyverse)
library(caret)

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

write.csv(cn_progress, 'data/mci_progress.csv')