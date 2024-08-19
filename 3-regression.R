df = read.csv("machineLearning.csv", row.names = NULL)
forced = df$score1 != df$choice1
interiorChoice1 = 0.1 <= df$choice1 & df$choice1 <= 0.4 

case = forced & interiorChoice1
summary(lm(df$choice2[case] ~ df$score1[case] + df$choice1[case]))
summary(lm(df$choice2[case] ~ df$score1[case] + df$prediction[case]))
print(sum(case))

case = forced & df$choice1 < 0.1
summary(lm(df$choice2[case] ~ df$score1[case] + df$choice1[case]))
summary(lm(df$choice2[case] ~ df$score1[case] + df$prediction[case]))
print(sum(case))

case = forced & df$choice1 > 0.4
summary(lm(df$choice2[case] ~ df$score1[case] + df$choice1[case]))
summary(lm(df$choice2[case] ~ df$score1[case] + df$prediction[case]))
print(sum(case))
plot(df$score1[case],df$choice2[case])