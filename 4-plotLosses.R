# Plot Losses

df = read.csv("loss.csv", row.names = NULL)

xlim = c(1,length(df$trainingLoss))
ymax = max(c(df$validationLoss))
ymin = min(c(df$validationLoss))
ylim = c(ymin,ymax)
plot(-1,xlim=xlim,ylim=ylim,xlab='Step',ylab='Validation Loss')
lines(df$validationLoss,col="blue")

# Find the optimal stopping point 
which.min(df$validationLoss)
