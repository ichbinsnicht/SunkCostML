library(plot.matrix)

values = as.matrix(read.csv('grid.csv',row.names = 1))
values[1:51,] = values[51:1,]
rownames(values) = seq(0.5,0.0,-0.01)
colnames(values) = seq(0.0,0.5,+0.01)

df = read.csv("machineLearning.csv", row.names = NULL)
plot(df$choice1,df$score1,pch=20,cex=4,col=rgb(0,1-2*df$choice2,2*df$choice2))

graphics.off()
par(mar=c(5.1, 4.1, 4.1, 4.1))
plot(values, border=NA, xlab='choice1', ylab='score1', col=rgb(0,seq(1,0,-0.1),seq(0,1,0.1)),main='Predicted Choice 2')