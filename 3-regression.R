df = read.csv('clean/cleanData.csv',row.names = NULL)
studies = c('66be300552fc3c98bcdc6c35','66c8d0c8779c5adea7f68954')
study1 = df$study == studies[1]
study2 = df$study == studies[2]

lowerBound = 0.1
upperBound = 0.4
interiorChoice1 = lowerBound <= df$choice1 & df$choice1 <= upperBound
lowChoice1 = df$choice1 < lowerBound
highChoice1 = df$choice1 > upperBound

summary(lm(choice2 ~ score1, data=df[interiorChoice1,]))
print(sum(interiorChoice1))

summary(lm(choice2 ~ score1, data=df[lowChoice1,]))
print(sum(lowChoice1))

summary(lm(choice2 ~ score1, data=df[highChoice1,]))
print(sum(highChoice1))

