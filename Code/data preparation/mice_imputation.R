library(readr)
library(dplyr)
library(ggplot2)
library(devtools)
library(mice)

print(getwd())
setwd('./R-Yoshimura_ALI')
path <- getwd()

"&" <- function(e1, e2) { # nolint
    if (is.character(c(e1, e2))) {
        paste(e1, e2, sep = "")
    } else {
        base::"&"(e1, e2) # nolint
    }
}

df_d1all_forMice <- read.csv(''&path&'/Data/df_forMice.csv', na.strings = c(" ", "NA","NULL"))

df_d1all_forMice <- df_d1all_forMice[,-2]

#colnames(df)
head(df_d1all_forMice)
 

m = 50
maxit = 50
df_d1all_forMice_mice <- mice(df_d1all_forMice,
                m = m,
                maxit = maxit,
                method = "pmm",
                seed = 1234)

for(i in 1:m){
    # 一個だけ取り出して保存
    miced_comp <- complete(df_d1all_forMice_mice, action = i)
    write.csv(miced_comp, file = ''&path&'/Data/miceData/df_afterMice_'&i&'.csv')
}
