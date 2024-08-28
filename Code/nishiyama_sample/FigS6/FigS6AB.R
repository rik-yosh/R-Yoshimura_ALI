library(readr)
library(dplyr)
library(ggplot2)
library(coin)
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

df <- readr::read_csv(""&path&"/data/ChanpionData/miceData/240214_df_afterMice_1.csv")  # nolint
y <- readr::read_csv(path&'/data/ChanpionData/estimatedIndividualParameters.txt')

y_true <- y[,c('g_mean','d_mean','P0_mean')]
y_true["type"] <- "Model fitting"
y_pred <- readr::read_csv(path&'/data/ChanpionData/addinfo/y_pred_LOO_RF_0.csv')
y_pred["type"] <- "RF prediction"

paraDensityPlt <- function(y_bind, c_type, c_paralist, colorlist, savefolder){
  plts <- list()
  for(i in 1:length(c_paralist)){
    print(c_paralist[i])
    x_param <- c_paralist[i]
    # x_param <- 'g_mean'
    strList <- str_split(x_param,"_")
    para <- strList[[1]][1]

    title<-para&" Distribution"
    xlabel<-para&" values"
    ylabel<-"Density"
    vlinesize <- 2
    vlinealpha <- 0.5

    data_use <- data.frame(
      value = y_bind[[x_param]],
      type = as.factor(y_bind[[c_type]])
    )
    
    uniqType <- unique(y_bind[[c_type]])
    plt_dur <- ggplot()+
      geom_density(data=data_use, aes(x=value, fill=type),alpha=0.3) +
      scale_fill_manual(values = colorlist)
    
    for(j in 1:length(uniqType)){
      plt_dur <- plt_dur + geom_vline(xintercept = mean(data_use[data_use$type==uniqType[[j]],]$value), col=colorlist[[j]], size=vlinesize, alpha=vlinealpha)
    }
    
    plt_dur <- plt_dur +
      ylab(ylabel)
    
    if(i==1){
      plt_dur <- plt_dur +
        theme(
        text = element_text(size = 24),axis.text = element_text(colour = "black"),
        plot.title = element_blank(),
        axis.ticks = element_line(colour = "black"),
        axis.line = element_line(colour = "black"),
        axis.title.x = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        legend.position = c(1, 1), legend.justification = c(1, 1)
        # legend.position='none'
      )
    }else{
      plt_dur <- plt_dur +
        theme(
        text = element_text(size = 24),axis.text = element_text(colour = "black"),
        plot.title = element_blank(),
        axis.ticks = element_line(colour = "black"),
        axis.line = element_line(colour = "black"),
        axis.title.x = element_blank(),
        panel.grid.major = element_blank(),
        panel.grid.minor = element_blank(),
        panel.background = element_blank(),
        # legend.position = c(1, 1), legend.justification = c(1, 1)
        legend.position='none'
        )
    }
    ggsave(plt_dur,file=savefolder&'/'&para&'_distribution.png',width=14,height=4.5,dpi=200)
    plts[[i]] <- plt_dur
  }
  return(plts)
}

y_bind <- rbind(y_true, y_pred)
c_type <- "type"
c_paralist <- c('g_mean','d_mean','P0_mean')
colorlist <- c("black","blue")

savefolder <- path&'/results/ChanpionData/paramComp'
plts <- paraDensityPlt(y_bind, c_type, c_paralist, colorlist, savefolder)


paraScatPlt <- function(y_true,y_pred, c_paralist, colorlist, savefolder){
  plts <- list()
  for(i in 1:length(c_paralist)){
    # i=1
    x_param <- c_paralist[i]
    # x_param <- 'g_mean'
    strList <- str_split(x_param,"_")
    para <- strList[[1]][1]
    
    data_use <- data.frame(
      True = y_true[[x_param]],
      Pred = y_pred[[x_param]]
    )
    lineData <- data.frame(
      x=c(min(c(data_use$True,data_use$Pred)),max(c(data_use$True,data_use$Pred))),
      y=c(min(c(data_use$True,data_use$Pred)),max(c(data_use$True,data_use$Pred)))
    )
    
    R2value <- 1-(sum((data_use$True - data_use$Pred)**2)/sum((data_use$True-mean(data_use$True))**2))
    
    xlabel<-"Estimated values by Fitting"
    ylabel<-"Predicted values by RF"
    R2_annotate <- round(R2value,2)

    plt_dur <- ggplot()+
      geom_point(data=data_use, aes(x=True, y = Pred),alpha=0.8,color="black") +
      geom_line(data = lineData, aes(x=x,y=y),color="red",linetype="dashed",show.legend = TRUE) +
      annotate("text",x=-Inf,y=Inf,label=bquote(R^2 ~ "=" ~ .(R2_annotate)),hjust=-.2,vjust=2, size=10) +
      xlab(xlabel) + ylab(ylabel)+
      #scale_x_continuous(expand = c(0, 0), breaks=seq(5,35,by=5), limits=c(3,37))+
      #scale_y_continuous(expand = c(0, 0), breaks=seq(0,0.15,by=0.05), limits=c(0,0.155))+
      theme(text = element_text(size = 24),axis.text = element_text(colour = "black"),
            plot.title = element_blank(),
            axis.ticks = element_line(colour = "black"),
            axis.line = element_line(colour = "black"),
            panel.grid.major = element_blank(),
            panel.grid.minor = element_blank(),
            panel.background = element_blank(),
            legend.position='none'
            )
    ggsave(plt_dur,file=savefolder&'/scatter_'&para&'_predtrue.png',width=5.5,height=5.5,dpi=200,)
    plts[[i]] <- plt_dur
  }
  return(plts)
}
plts <- paraScatPlt(y_true,y_pred, c_paralist, colorlist, savefolder)