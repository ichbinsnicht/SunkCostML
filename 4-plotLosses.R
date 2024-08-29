# Plot Losses

df = read.csv("loss.csv", row.names = NULL)

xlim = c(1,length(df$trainingLoss))
ymax = max(c(df$validationLoss))
ymin = min(c(df$validationLoss))
ylim = c(ymin,ymax)
plot(df$validationLoss,col="blue", type='l',xlab='Step',ylab='Validation Loss')

# Find the optimal stopping point 
which.min(df$validationLoss)
