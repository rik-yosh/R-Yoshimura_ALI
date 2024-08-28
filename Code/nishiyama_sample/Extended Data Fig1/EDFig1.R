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

# Make crosstable and Execute chi-square test.
# 
# @param row Factor vector.
# @param column Factor vector.
# @param data data.frame.
# @return table.
crosstab <- function(row, column, data=NULL){
    if(is.null(data)){
        cross.table <- table(row,column)
    }else{
        x <- substitute(row)
        y <- substitute(column)
        cross.table <- table(data[[x]],data[[y]])
    }
    names(dimnames(cross.table)) <- NULL
    class(cross.table) <- c("crosstab", "table")
    return(cross.table)
}

summary.crosstab <- function(object,upper=1000){
    upperN <- upper

    #CrossTable
    crossTable <- object
    class(crossTable) <- "table"
    crossTableMarin <- addmargins(crossTable)
    dim <- dim(crossTable)
    row <- dim[1]
    column <- dim[2]
    N <- sum(crossTable)
    df <- (nrow(crossTable) - 1)*(ncol(crossTable) - 1)

   if((df == 1) || min(crossTable)<=5){
        upperN <- 2000
   }

    #ratio
    row.ratio <- prop.table(crossTable,margin=1)
    column.ratio <- prop.table(crossTable,margin=2)

    #standardized residual
    expected.crossTable <- apply(crossTable,2,function(x){sum(x)*apply(crossTable,1,sum)/N})
    res.crossTable <- crossTable - expected.crossTable
    std.res.crossTable <- res.crossTable/sqrt(expected.crossTable)
    
    #Yates' correction
    yates.std.res.crossTable <- (abs(res.crossTable) - 0.5)/sqrt(expected.crossTable)
    yates.std.res.crossTable[which(yates.std.res.crossTable < 0)] <- 0

    #chisq
    chisq.value <- sum(std.res.crossTable^2)
    yates.chisq.value <- sum(yates.std.res.crossTable^2)

    #p
    p.value <- 1 - pchisq(chisq.value,df)
    yates.p.value <- 1 - pchisq(yates.chisq.value,df)
    if(N <= upperN){
        fisher <- fisher.test(crossTable, alternative = "two.sided", workspace=10000000,simulate.p.value=TRUE)
    }

    #chisq-test output
    if(df == 1 && N <= upperN){
        chi.result <- matrix(0,3,3)
    }else if(df == 1 && N > upperN || df > 1 && N <=upperN){
        chi.result <- matrix(0,2,3)
    }else{
        chi.result <- matrix(0,1,3)
    }
    chi.result[1,1] <- chisq.value
    chi.result[1,2] <- df
    chi.result[1,3] <- p.value
    if(df ==1){
        chi.result[2,1] <- yates.chisq.value
        chi.result[2,2] <- df
        chi.result[2,3] <- yates.p.value
        if(N <= upperN){
            chi.result[3,1] <- NA
            chi.result[3,2] <- NA
            chi.result[3,3] <- fisher[[1]]
        }
    }else if(N <= upperN){
        chi.result[2,1] <- NA
        chi.result[2,2] <- NA
        chi.result[2,3] <- fisher[[1]]
    }

    if(df == 1 && N <= upperN){
        rownames(chi.result) <- c("Peason","Yates","Fisher")
    }else if(df ==1 && N > upperN){
        rownames(chi.result) <- c("Peason","Yates")
    }else if(N <= upperN){
        rownames(chi.result) <- c("Peason","Fisher")
    }else{
        rownames(chi.result) <- c("Peason")
    }
    colnames(chi.result) <- c("chi sq","df","P")

    #residual analysis
    se.crossTable <- sqrt(apply(crossTable,2,function(x){(1 - sum(x)/N )*(1 - apply(crossTable,1,sum)/N)}))
    adj.res.crossTable <- std.res.crossTable/se.crossTable
    res.p.crossTable <- (1 - pnorm(abs(adj.res.crossTable)))*2
    res.p.crossTable <- round(res.p.crossTable, 4)
    
    #Cramer V
    Cramer <- matrix(0,1,1)
    Cramer[1,1] <- sqrt(chisq.value/(N*(min(nrow(crossTable),ncol(crossTable))-1)))
    rownames(Cramer) <- c("Cramer's V")
    
    result <- list(crossTable=crossTable,row.ratio=row.ratio,column.ratio=column.ratio,chisq.test=chi.result,residualAnalysis=adj.res.crossTable,res.p.value=res.p.crossTable,Cramer=Cramer)
    class(result) <- c("listMatrix")
    
    return(result)
}

library("scales")
library("stringr")

scientific_10 <- function(x) {
  index_zero <- which(x == 0)
  label <- scientific_format()(x)
  label <- str_replace(label, "e", " %*% 10^")
  label <- str_replace(label, "\\^\\+", "\\^")
  label[index_zero] <- "0"
  parse(text=label)
}

chi_resi <- function(df_cat,ClustLabel,dataName,csvSaveFolder,b_cross=TRUE,b_exp=TRUE,b_chisq=TRUE,b_rRat=TRUE,b_resi=TRUE,b_resiP=TRUE,b_cramV=TRUE){
  result_list <- list()
  cross = c()
  result = c()
  # データを抽出したデータフレームを作成
  cross <- data.frame(
      ClustLabel = df_cat[ClustLabel][[1]],
      data = df_cat[dataName][[1]]
  )
  cross$ClustLabel <- as.factor(cross$ClustLabel)

  #クロス集計パッケージでの処理
  crossTable <- crosstab(ClustLabel, data, cross)

  #χ2検定他
  result <- summary.crosstab(crossTable)
  ####################################
  # クロス集計表の保存
  ####################################
  #データフレーム化
  result_cross <- data.frame(result$crossTable) %>% pivot_wider(names_from = "Var2",values_from = "Freq")
  colnames(result_cross)[1] <- "ClustLabel"
  result_list[[1]] <- result_cross

  ####################################
  # 期待値表の保存
  ####################################
  result_exp <- result_cross

  Ncol <- length(colnames(result_exp))
  Nrow <- length(rownames(result_exp))
  rSum <- c()
  
  for(i in result_exp$ClustLabel){
      rSum <- append(rSum, sum(result_exp[result_exp$ClustLabel==i,2:Ncol]))
  }
  TotalSum <- sum(rSum)

  for(n in 1:Nrow){
      for(m in 2:Ncol){
          ClustSum <- rSum[[n]]
          dataSum <- sum(result_exp[,m])
          result_exp[n,m] <- round(dataSum*ClustSum/TotalSum,digits = 0)
      }
  }
  result_list[[2]] <- result_exp

  ####################################
  # chisqTest結果の保存
  ####################################
  #データフレーム化
  result_chisq <- data.frame(result$chisq.test)
  result_chisq['test'] <- rownames(result_chisq)
  result_list[[3]] <- result_chisq

  if((any(result_exp[,2:length(result_exp)]<5)&(result_chisq["Fisher","P"]<0.05))|(all(result_exp[,2:length(result_exp)]>=5)&(result_chisq["Peason","P"]<0.05))){
    if(b_cross){write.csv(result_cross, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_cross.csv", quote=FALSE, row.names=FALSE)}
    if(b_exp){write.csv(result_exp, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_exp.csv", quote=FALSE, row.names=FALSE)}
    if(b_chisq){write.csv(result_chisq, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_chisq.csv", quote=FALSE, row.names=FALSE)}
    ####################################
    # 行方向の比率データの保存
    ####################################
    #データフレーム化
    result_rRatio <- data.frame(result$row.ratio) %>% pivot_wider(names_from = "Var2",values_from = "Freq")
    colnames(result_rRatio)[1] <- "ClustLabel"
    result_list[[4]] <- result_rRatio
    if(b_rRat){write.csv(result_rRatio, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_rowRatio.csv", quote=FALSE, row.names=FALSE)}

    ####################################
    # 残差分析データの保存
    ####################################
    #データフレーム化
    result_resi <- data.frame(result$residualAnalysis)%>% pivot_wider(names_from = "Var2",values_from = "Freq")
    colnames(result_resi)[1] <- "ClustLabel"
    result_list[[5]] <- result_resi
    if(b_resi){write.csv(result_resi, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_resiVal.csv", quote=FALSE, row.names=FALSE)}

    ####################################
    # 残差分析p値の保存
    ####################################
    #データフレーム化
    result_resPval <- data.frame(result$res.p.value)
    result_list[[6]] <- result_resPval
    if(b_resiP){write.csv(result_resPval, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_resiPval.csv", quote=FALSE, row.names=FALSE)}

    ####################################
    # 効果量Cramer's Vの保存
    ####################################
    #データフレーム化
    result_Cramer <- data.frame(result$Cramer)
    result_list[[7]] <- result_Cramer
    if(b_cramV){write.csv(result_Cramer, ""&csvSaveFolder&"/chi_"&dataName_list[a]&"_CramerV.csv", quote=FALSE, row.names=FALSE)}
  }
  return(result_list)
}
ratioplot <- function(data,ClustLabel,dataName,result_list,legend=TRUE,pltSaveFolder){
  # データを抽出したデータフレームを作成
  cross <- data.frame(
      ClustLabel = data[ClustLabel][[1]],
      data = data[dataName][[1]]
  )
  cross$ClustLabel <- as.factor(cross$ClustLabel)

  result_exp <- result_list[[2]]
  result_chisq <- result_list[[3]]

  resPval <- resultlist[[6]]
  resPval <- resPval[resPval$Var2==TRUE,]
  star <-  ifelse(resPval$Freq >= .05, "NS.",ifelse(resPval$Freq >= 0.01, "＊", ifelse(resPval$Freq >= 0.001, "＊＊", "＊＊＊")))

  if(any(result_exp[,2:length(result_exp)]<=5)){
    if(result_chisq["Fisher","P"][[1]]<0.001){
      annotateText <- "Fisher: p<0.001"
    }else if (result_chisq["Fisher","P"][[1]] > 0.05) {
      annotateText <- "Fisher: N.S."
    }else{
      annotateText <- "Fisher: p="&round(result_chisq["Fisher","P"][[1]],4)&""
    }
  }else{
    if(result_chisq["Peason","P"][[1]]<0.001){
      annotateText <- "Peason: p<0.001"
    }else if (result_chisq["Peason","P"][[1]] > 0.05) {
      annotateText <- "Peason: N.S."
    }else{
      annotateText <- "Peason: p="&round(result_chisq["Peason","P"][[1]],4)&""
    }
  }
  

  ####################################
  # 割合をプロットで表す
  ####################################
  #プロット用データフレームの作成
  pltData <-  cross %>% 
              group_by(ClustLabel,data) %>% 
              summarise(count = n()) %>%
              mutate(prop = round(count/sum(count),digits = 3))
  write.csv(pltData, ''&path&'/data/ChanpionData/forTable/Categol/'&dataName&'.csv')
  #タイトルとかの設定
  title<-""&dataName&""
  xlabel<-"Group"
  ylabel<-""&dataName&""
  cols <- c(brewer.pal(12,'Set3'),brewer.pal(8,'Pastel2'),brewer.pal(9,'Pastel1'),brewer.pal(8,'Set2'))
  
  sx <- 5

  #帯グラフ
  plt_dur<- ggplot(data = pltData ,aes(x = ClustLabel,y = prop, fill = data,label = data)) +
    geom_bar(stat = "identity", position = "fill") +
    scale_y_continuous(labels = percent) +
    scale_fill_manual(values = cols)+
    #geom_text(aes(y=prop,size=8),position = position_stack(vjust = .25))+
    xlab(xlabel) + ylab(ylabel)+ggtitle(title)+ylim(0,1.05)+
    annotate("text",x=1,y=Inf,label=""&star[1]&"",size=sx,hjust=.5,vjust=1)+
    annotate("text",x=2,y=Inf,label=""&star[2]&"",size=sx,hjust=.5,vjust=1)+
    annotate("text",x=3,y=Inf,label=""&star[3]&"",size=sx,hjust=.5,vjust=1)+
    annotate("text",x=4,y=Inf,label=""&star[4]&"",size=sx,hjust=.5,vjust=1)+
    annotate("text",x=5,y=Inf,label=""&star[5]&"",size=sx,hjust=.5,vjust=1)+
    annotate("text",x=6,y=Inf,label=""&star[6]&"",size=sx,hjust=.5,vjust=1)
    # annotate("text",x=-Inf,y=Inf,label=""&annotateText&"",size=6,hjust=-.05,vjust=1)
  
  if(legend){
    plt_dur <- plt_dur +theme(
      text = element_text(size = 24),
      title =element_text(size=19),
      axis.text = element_text(colour = "black"),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      plot.title = element_text(colour = "black"),
      legend.text = element_text(),
      legend.title = element_text(),
      axis.ticks = element_line(colour = "black"),
      axis.line = element_line(colour = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      #legend.position='none'
      )
    ggsave(plt_dur,file=''&pltSaveFolder&'/EF2_categol_'&dataName&'.png',width=5,height=4,dpi=100)
  }else{
    plt_dur <- plt_dur +theme(
      text = element_text(size = 24),
      title =element_text(size=19),
      axis.text = element_text(colour = "black"),
      axis.title.x = element_blank(),
      axis.title.y = element_blank(),
      plot.title = element_text(colour = "black"),
      legend.text = element_text(),
      legend.title = element_text(),
      axis.ticks = element_line(colour = "black"),
      axis.line = element_line(colour = "black"),
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      panel.background = element_blank(),
      legend.position='none'
      )
    ggsave(plt_dur,file=''&pltSaveFolder&'/EF2_categol_'&dataName&'_nolegend.png',width=5,height=4,dpi=100)
  }
  return(plt_dur)
}
anova_bonf <- function(df_num,dataName_list,bonf_list){
  aov_list <- c()
  bonf_pcol <- c()
  for(i in 1:length(dataName_list)){
    # i=1
    # bonf_list <- tukey_list
    data_use <- df_num[which(df_num$numData == dataName_list[i]),]
    data_use$ClustLabel <- factor(data_use$ClustLabel)

    aov_result <- aov(value ~ ClustLabel, d = data_use)
    aov_pval <- as.numeric(summary(aov_result)[[1]]["ClustLabel","Pr(>F)"])

    aov <- c(as.numeric(aov_pval), aov_pval < 0.05)
    aov_list <- cbind(aov_list, aov)

    if(is.na(aov_pval)){
      print(i)
    }else {
      bonf_result <- pairwise.wilcox.test(data_use$value, data_use$ClustLabel, p.adj="bonferroni", exact=F)
      bonf_pval_DF <- bonf_result[[3]]
      bonf_pval <- bonf_pval_DF[!is.na(bonf_pval_DF)]
      bonf <- data.frame(pval = bonf_pval)
      bonf["psign"] <- ifelse(bonf$pval>0.05,"",ifelse(bonf$pval>0.01,"＊",ifelse(bonf$pval>0.001,"＊＊","＊＊＊")))
      if(sum(bonf[,"pval"] < 0.05) >= 1){
        bonf_pcol <- append(bonf_pcol,i)
      }
      names(bonf) <- c(dataName_list[i],"psign_"&dataName_list[i])
      bonf_list <- cbind(bonf_list, bonf)
    }
  }
  ### ANOVAのgtテーブル化
  aov_list <- as.data.frame(aov_list)
  rownames(aov_list) <- c('aov_pval','significant')
  colnames(aov_list) <- dataName_list

  ### Tukeyのgtテーブル化
  aov_pcol <- which(aov_list[rownames(aov_list)=='significant',] == 1)

  pcol <- intersect(bonf_pcol,aov_pcol)

  bonf_list <- data.frame(bonf_list)
  bonf_list <- bonf_list %>% separate(bonf_list, c("one", "another"), sep="-")

  aov_bonf <- list(aov_list,bonf_list,pcol)
  return(aov_bonf)
}
violinPerSuv <- function(data,numData="nan",ClustLabel="ClustLabel",value="value",bonf_list,title,xlabel,ylabel,cols,filename){
  data_use <- data.frame(
    ClustLabel = data[ClustLabel][[1]],
    values = data[value][[1]]
  )
  dname <- unique(data[numData][[1]])

  vmin <- min(data_use$values)
  vmax <- max(data_use$values)
  vrange <- vmax - vmin

  ann_psign <- wil_list[colnames(wil_list) == paste("psign_",dname, sep="")]

  plt_dur <- ggplot(data=data_use,aes(x=as.factor(ClustLabel), y=values))+
    geom_violin(aes(alpha=0.5, fill=as.factor(ClustLabel)))+# violinplot
    geom_boxplot(width = .1, fill ="white")+#箱ひげ図
    stat_summary(fun = "mean", geom = "point", shape = 23, size = 1.5, alpha=1, fill = "black") + #平均値も見せる
    geom_jitter(shape = 21, size = 1, width=0.2) +#実際のデータも一応プロット
    scale_fill_manual(values = cols) +# 色指定(UMAPカラーパレット)
    geom_signif(comparisons = list(c("non-TFS","TFS")),step_increase = 0.15,annotations = ann_psign,tip_length = 0.01,textsize = 4)+
    xlab(xlabel) +
    ylab(ylabel)+
    ggtitle(title)+
    scale_y_continuous(label=scientific_10)+
    # scale_y_continuous(limits=c(vmin-0.2*vrange,vmax+0.2*vrange))+
    theme(text = element_text(size = 24),
          title =element_text(size=19),
          axis.text = element_text(colour = "black"),
          axis.ticks = element_line(colour = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          legend.position='none')
  
  ggsave(plt_dur,file=filename,width=5,height=4,dpi=300)
  
  return(plt_dur)
}
violinPerSuv_PT <- function(data,numData="nan",ClustLabel="ClustLabel",value="value",bonf_list,title,xlabel,ylabel,cols,filename){
  data_use <- data.frame(
    ClustLabel = data[ClustLabel][[1]],
    values = data[value][[1]]
  )
  dname <- unique(data[numData][[1]])

  vmin <- min(data_use$values)
  vmax <- max(data_use$values)
  vrange <- vmax - vmin

  ann_psign <- wil_list[colnames(wil_list) == paste("psign_",dname, sep="")]

  plt_dur <- ggplot(data=data_use,aes(x=as.factor(ClustLabel), y=values))+
    geom_violin(aes(alpha=0.5, fill=as.factor(ClustLabel)))+# violinplot
    geom_boxplot(width = .1, fill ="white")+#箱ひげ図
    stat_summary(fun = "mean", geom = "point", shape = 23, size = 1.5, alpha=1, fill = "black") + #平均値も見せる
    geom_jitter(shape = 21, size = 1, width=0.2) +#実際のデータも一応プロット
    scale_fill_manual(values = cols) +# 色指定(UMAPカラーパレット)
    # geom_hline(yintercept = 50.29, col="#f0007e",linetype='dotted', size=2, alpha=0.9)+
    geom_signif(comparisons = list(c("non-TFS","TFS")),step_increase = 0.15,annotations = ann_psign,tip_length = 0.01,textsize = 4)+
    xlab(xlabel) +
    ylab(ylabel)+
    ggtitle(title)+
    scale_y_continuous(label=scientific_10)+
    # scale_y_continuous(limits=c(vmin-0.2*vrange,vmax+0.2*vrange))+
    theme(text = element_text(size = 24),
          title =element_text(size=19),
          axis.text = element_text(colour = "black"),
          axis.ticks = element_line(colour = "black"),
          axis.title.x = element_blank(),
          axis.title.y = element_blank(),
          axis.line = element_line(colour = "black"),
          panel.grid.major = element_blank(),
          panel.grid.minor = element_blank(),
          panel.background = element_blank(),
          legend.position='none')
  
  ggsave(plt_dur,file=filename,width=5,height=4,dpi=300)
  
  return(plt_dur)
}
violinPerClust <- function(data,numData="nan",ClustLabel="ClustLabel",value="value",bonf_list,title,xlabel,ylabel,cols,filename){
  data_use <- data.frame(
    ClustLabel = data[ClustLabel][[1]],
    values = data[value][[1]]
  )
  dname <- unique(data[numData][[1]])

  vmin <- min(data_use$values)
  vmax <- max(data_use$values)
  vrange <- vmax - vmin

  print(dname)
  bonf_list2 <- bonf_list[bonf_list["psign_"&dname][[1]]!="", ]
  ann_psign <- bonf_list2[,"psign_"&dname]
  
  if(length(row.names(bonf_list2)!=0)){
    complist = list()
    for(i in 1:length(row.names(bonf_list2))){
      complist[[i]] <- c(bonf_list2[i,"one"],bonf_list2[i,"another"])
    }
    plt_dur <- ggplot(data=data_use,aes(x=as.factor(ClustLabel), y=values))+
      geom_violin(aes(alpha=0.5, fill=as.factor(ClustLabel)))+# violinplot
      geom_boxplot(width = .1, fill ="white")+#箱ひげ図
      stat_summary(fun = "mean", geom = "point", shape = 23, size = 1.5, alpha=1, fill = "black") + #平均値も見せる
      geom_jitter(shape = 21, size = 1, width=0.2) +#実際のデータも一応プロット
      geom_signif(comparisons = complist,step_increase = 0.15,annotations = ann_psign,tip_length = 0.01,textsize = 4)+
      scale_fill_manual(values = cols) +# 色指定(UMAPカラーパレット)
      xlab(xlabel) + ylab(ylabel)+ggtitle(title)+
      scale_y_continuous(label=scientific_10)+
      # scale_y_continuous(limits=c(vmin-0.2*vrange,vmax+0.2*vrange))+
      theme(text = element_text(size = 24),axis.text = element_text(colour = "black"),
              title =element_text(size=19),
              axis.ticks = element_line(colour = "black"),
              axis.line = element_line(colour = "black"),
              axis.title.x = element_blank(),
              axis.title.y = element_blank(),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.background = element_blank(),
              legend.position='none')
    ggsave(plt_dur,file=filename,width=5,height=4,dpi=300)
  }else{
    plt_dur <- ggplot(data=data_use,aes(x=as.factor(ClustLabel), y=values))+
      geom_violin(aes(alpha=0.5, fill=as.factor(ClustLabel)))+# violinplot
      geom_boxplot(width = .1, fill ="white")+#箱ひげ図
      stat_summary(fun = "mean", geom = "point", shape = 23, size = 1.5, alpha=1, fill = "black") + #平均値も見せる
      geom_jitter(shape = 21, size = 1, width=0.2) +#実際のデータも一応プロット
      scale_fill_manual(values = cols) +# 色指定(UMAPカラーパレット)
      xlab(xlabel) + ylab(ylabel)+ggtitle(title)+
      scale_y_continuous(label=scientific_10)+
      # scale_y_continuous(limits=c(vmin-0.2*vrange,vmax+0.2*vrange))+
      theme(text = element_text(size = 24),axis.text = element_text(colour = "black"),
              title =element_text(size=19),
              axis.ticks = element_line(colour = "black"),
              axis.line = element_line(colour = "black"),
              axis.title.x = element_blank(),
              axis.title.y = element_blank(),
              panel.grid.major = element_blank(),
              panel.grid.minor = element_blank(),
              panel.background = element_blank(),
              legend.position='none')
    ggsave(plt_dur,file=filename,width=5,height=4,dpi=300)
  }
  return(plt_dur)
}

###############################################################################################################################################################
# 内科的生死でのプロット
###############################################################################################################################################################
## 準備
df <- readr::read_csv(""&path&"/data/ChanpionData/miceData/240214_df_afterMice_1.csv")  # nolint
LifeData = readr::read_csv(path&"/data/ChanpionData/dataloading/240214_df_d0all.csv")
life <- LifeData$InternalMedicineSurvival
# プロット時に表示をややこしくしないため
df = df %>% rename("FFPprescribing_U_d0" = Yesterday_FFPprescribing_U_d1) %>% rename("FFPprescribing_U_d1" = Yesterday_FFPprescribing_U_d2 )%>% rename("FFPprescribing_U_d2" = Yesterday_FFPprescribing_U_d3) %>% rename("FFPprescribing_U_d3" = Yesterday_FFPprescribing_U_d7)
clustData = readr::read_csv(""&path&"/data/ChanpionData/DTWclustLabel.csv")  # nolint

# day7のデータのみ
d7Feature = c('PT_percentage_d7','Plt_d7','Cre_d7','BUN_d7','ALT_d7','LDH_d7','Alb_d7','TDratio_d7','AST_d7')
df_num = df[,d7Feature]
df_num$ClustLabel = clustData$dtw_6

IMS <- c()
for(l in life){
  IMS <- append(IMS,ifelse(l==1,"non-TFS","TFS"))
}

df_num$IMS <- IMS

## 数値データのANOVA
df_num = df_num %>% pivot_longer(cols = d7Feature,names_to = "numData",values_to = "value")
dataName_list <- c((unique(df_num$numData)))


wil_list <- c("TFS","non-TFS")
colN_list <- c("one","another")
for(i in 1:length(dataName_list)){
  # i = 1
  # wilcoxの検定を行う
  data_use <- df_num[which(df_num$numData == dataName_list[i]),]
  data_use <- data.frame(
    ClustLabel = data_use$IMS,
    values = data_use$value
  )
  # Male p-val data
  wil_test <- wilcox.exact(x=data_use[data_use$ClustLabel == "non-TFS",]$values,y=data_use[data_use$ClustLabel == "TFS",]$values,paired=F)
  p_val <- wil_test$p.value
  star <-  ifelse(p_val >= .05, "NS.",ifelse(p_val >= 0.01, "＊", ifelse(p_val >= 0.001, "＊＊", "＊＊＊")))
  wil_list <- append(wil_list,p_val)
  wil_list <- append(wil_list,star)

  # Make columns name
  colN_list <- append(colN_list, dataName_list[i])
  colN_list <- append(colN_list, "psign_"&dataName_list[i])
}
wil_list = t(data.frame(wil_list))
colnames(wil_list) <- colN_list
# vilolin plot

data <- vector()
plts = list()
# プロット準備
for(i in 1:length(dataName_list)){ 
  # i = 1
  data <- df_num[which(df_num$numData == dataName_list[i]),] #有意差のあるデータを一つずつ取り出す
  data <- transform(data, ClustLabel= factor(ClustLabel, levels = c("G1","G2","G3","G4","G5","G6","G7","G8","G9","G10")))#表示のためのレベルの設定

  namesplit <- str_split(dataName_list[i], "_")
  if(namesplit[[1]][1]=="TDratio"){
    title<-"D/T-bil"
  }else if (namesplit[[1]][1]=="PT") {
     title<-"PT%"
  }
  else{
    title<-""&namesplit[[1]][1]&""
  }
  xlabel<-"Survival in Internal Medicine"
  ylabel<-"Value"
  cols <- c("#fdcdac","#b3e2cd")
  
  ClustLabel="IMS"
  value = "value"
  numData = "numData"
  filename = ''&path&'/results/ChanpionData/ExtendedFig/EF1_Violonplot_'&dataName_list[i]&'.pdf'
  if(dataName_list[i] == "PT_percentage_d7"){
    plt = violinPerSuv_PT(data,numData,ClustLabel,value,wil_list,title,xlabel,ylabel,cols,filename)
  }else{
    plt = violinPerSuv(data,numData,ClustLabel,value,wil_list,title,xlabel,ylabel,cols,filename)
  }
  plts[[i]] <- plt
}

ggGrid <- grid.arrange(grobs = plts, ncol = 3)
filename = ''&path&'/results/ChanpionData/ExtendedFig/EF1_gridExtra_Violonplot.png'
ggsave(ggGrid,file=filename,width=15,height=12,dpi=300)

 

###############################################################################################################################################################
# グループごとのプロット
###############################################################################################################################################################
## 準備
df <- readr::read_csv(""&path&"/data/ChanpionData/ExFig2_day0_rename.csv")  # nolint
colnames(df)
# プロット時に表示をややこしくしないため
clustData = readr::read_csv(""&path&"/data/ChanpionData/DTWclustLabel.csv")  # nolint

d0Feature_num = c('ATIII','Che','PT','AST','gGTP','ALT','LDH','APTT','NH3')
d0Feature_cat = c('No LA')

## 数値データ
df_num = df[,d0Feature_num]
df_num$ClustLabel = clustData$dtw_6
df_num = df_num %>% pivot_longer(cols = d0Feature_num,
                                names_to = "numData",values_to = "value")
dataName_list <- unique(df_num$numData)

tukey_list <- c("G1-G2","G1-G3","G1-G4","G1-G5","G1-G6","G2-G3","G2-G4","G2-G5","G2-G6","G3-G4","G3-G5","G3-G6","G4-G5","G4-G6","G5-G6")
anova_bonf_list <- anova_bonf(df_num,dataName_list,tukey_list)

aov_list <- anova_bonf_list[[1]]
bonf_list <- anova_bonf_list[[2]]
pcol <- anova_bonf_list[[3]]


### Vilolin Plotの作成
# vilolin plot
data_use <- vector()
plts = list()
# プロット準備
for(i in 1:length(dataName_list)){
  # i = 1
  data_use <- df_num[which(df_num$numData == dataName_list[i]),] #有意差のあるデータを一つずつ取り出す
  data_use <- transform(data_use, ClustLabel= factor(ClustLabel, levels = c("G1","G2","G3","G4","G5","G6","G7","G8","G9","G10")))#表示のためのレベルの設定

  if(dataName_list[i]=="PT"){
   title<-"PT%" 
  }else{
    title<-""&dataName_list[i]&""
  }
  xlabel<-"Group"
  ylabel<-"Value"
  cols <- c("#8c564b","#d62728","#ff7f0e","#1f77b4","#2ca02c","#9467bd","#e377c2","#7f7f7f","#bcbd22","#17becf")#  UMAPの色合いと合わせるためのカラーパレット
  
  ClustLabel="ClustLabel"
  value = "value"
  numData = "numData"
  filename = ''&path&'/results/ChanpionData/ExtendedFig/EF2_Violonplot_'&dataName_list[i]&'.png'

  plt = violinPerClust(data_use,numData,ClustLabel,value,bonf_list,title,xlabel,ylabel,cols,filename)
  plts[[i]] <- plt
}
ggGrid <- grid.arrange(grobs = plts, ncol = 3)
filename = ''&path&'/results/ChanpionData/ExtendedFig/EF2_gridExtra.png'
ggsave(ggGrid,file=filename,width=15,height=12,dpi=300)


## カテゴリカルデータのプロット
df_cat = df[,d0Feature_cat]
df_cat$ClustLabel = clustData$dtw_6

dataName_list_cat <- unique(colnames(df_cat))

csvSaveFolder = ""&path&"/data/ChanpionData/ExtendedFig"
pltSaveFolder = ''&path&'/results/ChanpionData/ExtendedFig'
ClustLabel <- "ClustLabel"

chisq_dfs <- data.frame()
for(a in 1:length(d0Feature_cat)){
  # a = 1
  print("now:"&dataName_list_cat[a]&"")
  resultlist <- chi_resi(df_cat,ClustLabel,dataName_list_cat[a],csvSaveFolder,b_cross=FALSE,b_exp=FALSE,b_chisq=FALSE,b_rRat=FALSE,b_resi=FALSE,b_resiP=FALSE,b_cramV=FALSE)
  
  chisq_df <-  resultlist[[3]]
  chisq_df["dtype"] <- dataName_list_cat[a]
  chisq_dfs <- rbind(chisq_dfs,chisq_df[,c(3,4,5)])

  plts[[length(dataName_list)+a]] <- ratioplot(df_cat,ClustLabel,dataName_list_cat[a],resultlist,legend=FALSE,pltSaveFolder)
  ratioplot(df_cat,ClustLabel,dataName_list_cat[a],resultlist,legend=TRUE,pltSaveFolder)
}
chisq_dfs

ggGrid <- grid.arrange(grobs = plts, ncol = 3)
filename = ''&path&'/results/ChanpionData/ExtendedFig/EF2_gridExtra.png'
ggsave(ggGrid,file=filename,width=15,height=17,dpi=300)



