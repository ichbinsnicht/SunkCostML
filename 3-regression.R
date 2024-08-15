# Regressions
# Date: 2024-08-02

df = read.csv("machineLearning.csv", row.names = NULL)

summary(lm(choice2 ~ score1 + prediction,data=df))
summary(lm(choice2 ~ score1 + choice1,data=df))

# I)
# score1       0.03162    0.19929   0.159  0.87554   
# prediction   1.02435    0.28976   3.535  0.00208 **

# II)


# Simulation to understand difference above
n=23
x = rnorm(n)
z = rnorm(n)
w = rnorm(n)
eps = rnorm(n)
y = x*1+z+100*w+eps
summary(lm(y ~ x + z))
summary(lm(y ~ x + z + w))