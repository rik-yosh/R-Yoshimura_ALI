library(readr)
library(dplyr)
library(ggplot2)
library(exactRankTests)
library(tidyverse)
library(ggsignif)
library(psych)
library(gridExtra)
library(ggpubr)
library(rstatix)
library(corrplot)
library(scales)
# library(ggsci)
library(multcomp)
library(makedummies)
library(RColorBrewer)
library(janitor)
library(devtools)
# library(NSM3)
library(chisq.posthoc.test)

library(drc)
library(base)

"&" <- function(e1, e2) { # nolint
    if (is.character(c(e1, e2))) {
        paste(e1, e2, sep = "")
    } else {
        base::"&"(e1, e2) # nolint
    }
}

setwd('./1_ALF_new')
path <- getwd()
print(path)

# drcパッケージの4PLフィッティング
# 推定パラメータから推定線を書くためにのDFを作る
drc_est4PLDF <-function(b,c,d,e,dSize,xmin,xmax){
    x_seq <- seq(xmin,xmax,by=abs(xmax-xmin)/dSize)
    y_seq <- c()
    for(xi in x_seq){
        yi <- c + (d-c)/(1+exp(b*(xi-e)))
        y_seq <- append(y_seq,yi)
    }
    caldata <- data.frame(x_data=x_seq,y_data=y_seq)
    return(caldata)
}

# ロジスティック関数フィッティング
drc_est4PLPlot <- function(# ただし、セグメントは一つ。関数はestlineDFのみ対応。
    data,# 解析に使うデータ。ただし、x=time,y=sである必要がある
    dSize,# プロットする直線のx軸の範囲を何当分するか。
    confratio,# 信頼区間の割合（ex. 0.95など）
    xAxisName,# x軸名
    yAxisName,# y軸名
    pltFileName,# グラフを保存する場所を選ぶ
    csvParamName# 推定したパラメータのcsvを保存する場所を選ぶ
    ){
    # フィッティング
    fit_lr <- drm(y_data ~ x_data, data=data, fct=L.4())
    b <- summary(fit_lr)$coefficient['b:(Intercept)','Estimate'];c <- summary(fit_lr)$coefficient['c:(Intercept)','Estimate'];d <- summary(fit_lr)$coefficient['d:(Intercept)','Estimate'];e <- summary(fit_lr)$coefficient['e:(Intercept)','Estimate']
    b_se <- summary(fit_lr)$coefficient['b:(Intercept)','Std. Error'];c_se <- summary(fit_lr)$coefficient['c:(Intercept)','Std. Error'];d_se <- summary(fit_lr)$coefficient['d:(Intercept)','Std. Error'];e_se <- summary(fit_lr)$coefficient['e:(Intercept)','Std. Error']
    # 残差を計算する
    
    res_seq <- c()
    for(i in 1:length(data$x_data)){
        xi <- data$x_data[i]
        yi <- data$y_data[i]
        pred <- c + (d-c)/(1+exp(b*(xi-e)))
        res_i <- (yi - pred)^2
        res_seq <- append(res_seq,res_i)
    }
    # 決定係数とbicを求める
    R2 <- (1 - sum(res_seq)/sum((data$y_data-mean(data$y_data))^2))
    bic <- BIC(fit_lr)

    # 推定したパラメータからデータ再構築
    xmin <- min(data$x_data)
    xmax <- max(data$x_data)
    estline <- drc_est4PLDF(b,c,d,e,dSize,xmin,xmax)

    paramDataFrame <- data.frame(b = b,c = c,d = d,e = e,b_se = b_se,c_se = c_se,d_se = d_se,e_se = e_se)
    write.csv(paramDataFrame,csvParamName,row.names=FALSE)

    # 95%信頼水準での信頼係数を求める
    conf_coef <- qnorm((1 + confratio) / 2)
    # 信頼区間を作成する
    errDF <- data.frame(
        x_data = estline$x_data,
        conf.low = estline$y_data - (conf_coef*sd(resid(fit_lr))),
        conf.high = estline$y_data + (conf_coef*sd(resid(fit_lr)))
    )
    R2_digi2 <- round(R2,digits = 2)
    
    dataSuv = data[data$outcome==0,]
    dataDth = data[data$outcome==1,]

    plt <- ggplot() + # nolint
        geom_line(data=estline, aes(x = x_data, y = y_data),color='black')+
        geom_ribbon(data=errDF, fill="gray", alpha=0.5, aes(x=x_data, ymin=conf.low, ymax=conf.high))+
        geom_point(data=dataSuv, aes(x = x_data, y = y_data),color="#b3e2cd") +
        geom_point(data=dataDth, aes(x = x_data, y = y_data),color="#fdcdac") +
        # geom_vline(xintercept = m, color='magenta')+
        #annotate("text", x=min(data$PT_percentage), y=max(errDF$conf.high)*0.9,hjust =0,vjust=0.8,size=3,label="b: "&round(b,digits = 2)&" sd"&round(b_se,digits = 2)&"\nc: "&round(c,digits = 2)&" sd"&round(c_se,digits = 2)&"\nd: "&round(d,digits = 2)&" sd"&round(d_se,digits = 2)&"\ne: "&round(e,digits =2)&" sd"&round(e_se,digits = 2)&"\nR2: "&round(R2,digits = 3)&"\nBIC: "&round(bic,digits = 3)&"")+
        # annotate("text", x=2.0, y=min(estline$s),hjust =0,vjust=0,size=6,label=bquote(paste({R^2},"= ",.(R2_digi2),sep = "")))+
        # annotate("text", x=m, y=min(errDF$conf.low),hjust =0,vjust=0.5,color='magenta',size=6,label="TV= "&round(m,digits = 2)&"")+
        #scale_color_brewer(palette = "Set2") +
        xlab(xAxisName)+
        ylab(yAxisName)+
        #scale_x_continuous(expand = c(0, 0), breaks=seq(0,7,by=1), limits=c(0,8))+
        pltTheme
    ggsave(pltFileName, plot = plt, dpi = 200, width = 4.5, height = 4) # nolint
    return(plt)
}

pltTheme <- theme(
    plot.title = element_blank(),
    axis.text.x = element_text(size = 12,angle = 0, hjust =0.5, colour = "black"),
    axis.text.y = element_text(size = 12,angle = 0, hjust =0.5, colour = "black"),
    axis.ticks = element_line(colour = "black"),
    axis.line = element_line(colour = "black"),
    panel.grid.major = element_blank(),
    panel.grid.minor = element_blank(),
    panel.background = element_blank(),
    legend.position='none',
    axis.title.y = element_text(size=16,family="Helvetica"),
    axis.title.x = element_text(size=16,family="Helvetica")
    )

## 準備
ori_data <- readr::read_csv(path&"/data/ChanpionData/dataloading/240214_df_TS_Long.csv")  # nolint
supDF <- readr::read_csv(path&"/data/ChanpionData/dataloading/240214_df_d0all_dummy.csv")  # nolint

sup_list <- c()
for(i in 1:length(rownames(supDF))){
    sup_x <- rep(supDF$InternalMedicineSurvival[i],5)
    sup_list <- append(sup_list,sup_x)
}

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PTp vs PTINR
df <- data.frame(
    outcome = sup_list,
    x_data = ori_data$PT_percentage,
    y_data = ori_data$PTINR)
df <- na.omit(df)
head(df)

library(minerva)
library(corrplot)
resCor <- cor(df[,c("x_data","y_data")])
print("Pearson")
cor.test(df$x_data,df$y_data)
print("Spearman")
cor.test(df$x_data,df$y_data,method = "spearman")
print("MIC")
mine(df$x_data,df$y_data)$MIC

pdf(file = path&"/results/ChanpionData/changePT/corrplot_INR.pdf")
corrplot(resCor,addCoef.col="black")
dev.off()


confratio <- 0.95
dSize <-100
xAxisName <- "Prothrombin time activity percentage (%)"
yAxisName <- "Prothrombin time INR"
plotPATH <- ""&path&"/results/ChanpionData/changePT/regression_fit"
paraCsvPATH <- ""&path&"/data/ChanpionData/changePT/estParamater"

id_list <- unique(df$id)
p_list <- vector(mode = "list",length = length(id_list))
param <- c()
i=1
plt <- drc_est4PLPlot(
    data= df,
    dSize = dSize,
    confratio=confratio,
    xAxisName=xAxisName,
    yAxisName=yAxisName,
    pltFileName=""&plotPATH&"/fit_INR.png",
    csvParamName=""&paraCsvPATH&"/param_INR.csv"
    )
ggsave(file = plotPATH&"/fit_INR.png", plot = plt, dpi = 300, width = 4.8, height = 4.8)

param <- readr::read_csv(path&"/data/ChanpionData/changePT/estParamater/param_INR.csv")
pred <- param$c + (param$d-param$c)/(1+exp(param$b*(50.29-param$e)))
pred

#-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
# PTp vs PTs
df <- data.frame(
    outcome = sup_list,
    x_data = ori_data$PT_percentage,
    y_data = ori_data$PTs)
df <- na.omit(df)
head(df)

library(minerva)
library(corrplot)
resCor <- cor(df[,c("x_data","y_data")])
print("Pearson")
cor.test(df$x_data,df$y_data)
print("Spearman")
cor.test(df$x_data,df$y_data,method = "spearman")
print("MIC")
mine(df$x_data,df$y_data)$MIC

pdf(file = path&"/results/ChanpionData/changePT/corrplot_PTs.pdf")
corrplot(resCor,addCoef.col="black")
dev.off()


confratio <- 0.95
dSize <-100
xAxisName <- "Prothrombin time activity percentage (%)"
yAxisName <- "Prothrombin time (s)"
plotPATH <- ""&path&"/results/ChanpionData/changePT/regression_fit"
paraCsvPATH <- ""&path&"/data/ChanpionData/changePT/estParamater"


p_list <- vector(mode = "list",length = length(id_list))

plt <- drc_est4PLPlot(
    data= df,
    dSize = dSize,
    confratio=confratio,
    xAxisName=xAxisName,
    yAxisName=yAxisName,
    pltFileName=""&plotPATH&"/fit_PTs.png",
    csvParamName=""&paraCsvPATH&"/param_PTs.csv"
    )
ggsave(file = plotPATH&"/fit_PTs.png", plot = plt, dpi = 300, width = 4.8, height = 4.8)

param <- readr::read_csv(path&"/data/ChanpionData/changePT/estParamater/param_PTs.csv")
pred <- param$c + (param$d-param$c)/(1+exp(param$b*(50.29-param$e)))
pred
