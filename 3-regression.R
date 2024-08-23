df = read.csv('clean/cleanData.csv',row.names = NULL)
studies = c('66be300552fc3c98bcdc6c35','66c8d0c8779c5adea7f68954')
df = df[df$study==studies[2],]

lowerBound = 0.1
upperBound = 0.4
interiorChoice1 = lowerBound <= df$choice1 & df$choice1 <= upperBound

summary(lm(choice2 ~ score1, data=df[interiorChoice1,]))
print(sum(interiorChoice1))

summary(lm(choice2 ~ score1, data=df[df$choice1 < lowerBound,]))
print(sum(df$choice1 < lowerBound))

summary(lm(choice2 ~ score1, data=df[df$choice1 > upperBound,]))
print(sum(df$choice1 > upperBound))

