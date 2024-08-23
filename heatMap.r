library(plot.matrix)

values = as.matrix(read.csv('grid.csv',row.names = 1))
values[1:51,] = values[51:1,]
rownames(values) = seq(0.5,0.0,-0.01)
colnames(values) = seq(0.0,0.5,+0.01)

studies = c('66be300552fc3c98bcdc6c35','66c8d0c8779c5adea7f68954')
df = read.csv("clean/cleanData.csv", row.names = NULL)
df1 = df[df$study==studies[1],]
df2 = df[df$study==studies[2],]
plot(df1$choice1,df1$score1,pch=20,cex=6,col=rgb(0,1-2*df1$choice2,2*df1$choice2,0.5))
plot(df2$choice1,df2$score1,pch=20,cex=6,col=rgb(0,1-2*df2$choice2,2*df2$choice2,0.5))
plot(df$choice1,df$score1,pch=20,cex=6,col=rgb(0,1-2*df$choice2,2*df$choice2,0.5))

graphics.off()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(values, border=NA, xlab='choice1', ylab='score1', col=rgb(0,seq(1,0,-0.1),seq(0,1,0.1)),main='Predicted Choice 2')