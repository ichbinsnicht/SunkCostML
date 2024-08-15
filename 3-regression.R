# Regressions
# Date: 2024-08-02

df = read.csv("machineLearning.csv", row.names = NULL)

summary(lm(choice2 ~ score1 + choice1,data=df))
summary(lm(choice2 ~ score1 + prediction,data=df))