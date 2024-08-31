df = read.csv('clean/cleanData.csv',row.names = NULL)
studies = c('66be300552fc3c98bcdc6c35','66c8d0c8779c5adea7f68954','66d1edb1e75efcf8a6eeeb0c')
study1 = df$study == studies[1]
study2 = df$study == studies[2]
study3 = df$study == studies[3]

lowerBound = 0.1
upperBound = 0.4

interiorChoice1 = lowerBound <= df$choice1 & df$choice1 <= upperBound
lowChoice1 = df$choice1 < lowerBound
highChoice1 = df$choice1 > upperBound

interiorScore1 = lowerBound <= df$score1 & df$score1 <= upperBound
lowScore1 = df$score1 < lowerBound
highScore1 = df$score1 > upperBound

summary(lm(choice2 ~ score1, data=df[lowChoice1,]))
print(sum(lowChoice1))

summary(lm(choice2 ~ score1, data=df[interiorChoice1,]))
print(sum(interiorChoice1))

summary(lm(choice2 ~ score1, data=df[highChoice1,]))
print(sum(highChoice1))

plot(df$score1[lowChoice1],df$choice2[lowChoice1])
plot(df$score1[interiorChoice1],df$choice2[interiorChoice1])
plot(df$score1[highChoice1],df$choice2[highChoice1])
