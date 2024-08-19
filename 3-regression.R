df = read.csv('machineLearning.csv',row.names = NULL)
interiorChoice1 = 0.1 <= df$choice1 & df$choice1 <= 0.4 

summary(lm(choice2 ~ score1, data=df[interiorChoice1,]))
summary(lm(choice2 ~ score1 + prediction,data=df[interiorChoice1,]))
print(sum(interiorChoice1))

summary(lm(choice2 ~ score1, data=df[df$choice1 < 0.1,]))
summary(lm(choice2 ~ score1 + prediction,data=df[df$choice1 < 0.1,]))
print(sum(df$choice1 < 0.1))

summary(lm(choice2 ~ score1, data=df[df$choice1 > 0.4,]))
summary(lm(choice2 ~ score1 + prediction,data=df[df$choice1 > 0.4,]))
print(sum(df$choice1 > 0.4))
