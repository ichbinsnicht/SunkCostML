# Plot Losses
# Date: 2024-08-02

df = read.csv("loss.csv", row.names = NULL)

xlim = c(1,length(df$trainingLoss))
ylim = c(0,.5)
plot(-1,xlim=xlim,ylim=ylim)
lines(df$trainingLoss,col="blue")
lines(df$validationLoss,col="orange")

# Find the optimal stopping point 
which.min(df$validationLoss)
# 163 106
