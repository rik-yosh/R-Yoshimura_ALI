library(readr)
library(dplyr)
library(ggplot2)
library(tidyr)
library(data.table)

setwd('./R-Yoshimura_ALI')
path <- getwd()
print(path)

"&" <- function(e1, e2) { # nolint
    if (is.character(c(e1, e2))) {
        paste(e1, e2, sep = "")
    } else {
        base::"&"(e1, e2) # nolint
    }
}


# Data loading and preprocessing
dat <- fread(""&path&"/Data/Fig3/RMSEchange.csv",data.table = FALSE)
head(dat)
dat <- dat[,-c(3,4)]
head(dat)
dat <- dat %>% pivot_wider(names_from = d_num, values_from = RMSE_EP)
dat <- data.frame(dat)

head(dat)
colnames(dat) <- c("id_again","Clustering","d0","d0m","d1","d1m","d2","d2m","d3","d3m","d7","d7m")
head(dat)
write.csv(dat, ""&path&"/Data/Fig3/RMSEchangge_wider.csv", quote=FALSE, row.names=FALSE)