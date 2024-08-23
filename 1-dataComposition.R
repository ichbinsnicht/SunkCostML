# Data Composition
# Date: 2024-07-08

filePath = "raw/"
files = list.files(filePath)
dataFiles = files[grepl("data",files)]
preSurveyFiles = files[grepl("preSurvey",files)]
prolificFiles = files[grepl("prolific",files)]

dataFrames = list()
preSurveyFrames = list()
prolificFrames = list()

for (i in 1:length(dataFiles)){
  file = dataFiles[i]
  dataFile = paste0(filePath,file)
  print(dataFile)
  dataFrames[[i]] = read.csv(dataFile, row.names = NULL)
}
for (i in 1:length(preSurveyFiles)){
  file = preSurveyFiles[i]
  preSurveyFile = paste0(filePath,file)
  print(preSurveyFile)
  preSurveyFrames[[i]] = read.csv(preSurveyFile, row.names = NULL)
}
for (i in 1:length(prolificFiles)){
  file = prolificFiles[i]
  prolificFile = paste0(filePath,file)
  print(prolificFile)
  prolificFrames[[i]] = read.csv(prolificFile, row.names = NULL)
}
dataFrame = do.call("rbind",dataFrames)
preSurveyFrame = do.call("rbind",preSurveyFrames)
prolificFrame = do.call("rbind",prolificFrames)

rm(dataFrames, preSurveyFrames, prolificFrames)

dataFrame = dataFrame[dataFrame$practice==0,]
dataFrame = dataFrame[!duplicated(dataFrame$id),]

rownames(dataFrame) = dataFrame$id
rownames(preSurveyFrame) = preSurveyFrame$id
rownames(prolificFrame) = prolificFrame$Participant.id

ids = sort(dataFrame$id)

## Merging
dataFrame = dataFrame[ids,]
preSurveyFrame = preSurveyFrame[ids,]
prolificFrame = prolificFrame[ids,]

dataFrame = dataFrame[,c("study","score1","choice1","choice2")]
preSurveyFrame = preSurveyFrame[,-1]
prolificFrame = prolificFrame[,c(9,11:15,21)]

cleanFrame = cbind(preSurveyFrame,prolificFrame,dataFrame)
year = cleanFrame$Undergraduate.year.of.study
yearNumber = as.numeric(gsub("[^0-9.-]", "", year))
cleanFrame$Undergraduate.year.of.study = yearNumber

cleanFrame$Male = (cleanFrame$Sex == 'Male')*1
cleanFrame$Sex = NULL

cleanFrame$cognition1 = (cleanFrame$cognition1 == 5)*1
cleanFrame$cognition2 = (cleanFrame$cognition2 == 20)*1

cleanFrame$White = (cleanFrame$Ethnicity.simplified == 'White')*1
cleanFrame$Ethnicity.simplified = NULL

cleanFrame$Employment = (cleanFrame$Employment.status %in% c('Part-Time','Full-Time'))*1
cleanFrame$Employment.status = NULL


# Saving
write.csv(cleanFrame,"clean/cleanData.csv", row.names=FALSE)