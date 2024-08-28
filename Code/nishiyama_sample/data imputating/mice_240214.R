library(readr)
library(dplyr)
library(ggplot2)
library(devtools)
library(mice)
library(parallel)
library(doParallel)

print(getwd())
setwd('./1_ALF_new')
path <- getwd()

"&" <- function(e1, e2) { # nolint
    if (is.character(c(e1, e2))) {
        paste(e1, e2, sep = "")
    } else {
        base::"&"(e1, e2) # nolint
    }
}

df_d1all_forMice <- read.csv(''&path&'/data/ChanpionData/dataloading/240214_df_forMice.csv', na.strings = c(" ", "NA","NULL"))

df_d1all_forMice <- df_d1all_forMice[,-2]

#colnames(df)
head(df_d1all_forMice)
 

# cores <- detectCores(logical = FALSE)
# registerDoParallel(cores = 15)

# cl <- makeCluster(cores)
# clusterSetRNGStream(cl, 9956)
# clusterExport(cl, "df_d1all_forMice")
# clusterEvalQ(cl, library(mice))
# imp_pars <- 
#   parLapply(cl = cl, X = 1:cores, fun = function(no){
#     mice(df_d1all_forMice,
#                 m = 50,
#                 maxit = 50,
#                 method = "pmm",
#                 seed = 1234)
#   })
# stopCluster(cl)

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
    write.csv(miced_comp, file = ''&path&'/data/ChanpionData/miceData/240214_df_afterMice_'&i&'.csv')
}
