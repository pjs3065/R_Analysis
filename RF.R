#install library
class_pack = c("caret", "skimr", "RANN","randomForest","fastAdaboost","gbm","xgboost", "caretEnsemble","C50","earth")
install.packages(class_pack)

#include library
library(caret)
library(skimr)
library(RANN)

#Load url dataset
url_address = 'https://www.dropbox.com/s/4wpkhme7476zdt3/ dataset.csv?dl=1'
dataset <- read.csv(url_address)

#divide training testing
str(dataset)
trainRowNumbers = createDataPartition(dataset$class, p = 0.8, list = FALSE)

#create train, test dataset
trainData = dataset[trainRowNumbers,]
testData = dataset[-trainRowNumbers,]

# x coordinate / y coordinate
x = trainData[,1:20]
y = trainData$class

#check N/A
anyNA(trainData)
anyNA(testData)

#hot encording
preProcess_missingdata_model = preProcess(trainData, method = 'knnImpute')
trainData_impute = predict(preProcess_missingdata_model,newdata = trainData)
dummies_model = dummyVars(class ~., data = trainData_impute)
trainData_mat = predict(dummies_model, newdata = trainData_impute)
trainData_dummy = data.frame(trainData_mat)

#range
preProcess_range_model = preProcess(trainData_dummy, method = 'range')
trainData_pre = predict(preProcess_range_model, newdata = trainData_dummy)
apply(trainData_pre[, 1:61], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})
trainData_pre$class = y 

#traindata trainning
model_rf = train(class ~., data = trainData_pre, method = 'rf')
model_rf

#view plot
plot(model_rf)
varimp_rf = varImp(model_rf)
plot(varimp_rf)

#work test dataset
testData2 = predict(preProcess_missingdata_model,testData)
testData3 = predict(dummies_model, testData2)
testData3 = data.frame(testData3)
testData4 = predict(preProcess_range_model, testData3)

# Hyper parameter tuning
fitControl <- trainControl( method = 'cv', number = 8, savePredictions = 'final', classProbs = T, summaryFunction = twoClassSummary)

# training earth/rf/svmRadial/nnet
model_mars5 = train(class ~., data = trainData_pre, method = 'pcaNNet', metric = 'ROC', tuneLength = 5, trControl = fitControl)

#predict
predicted_rf5 = predict(model_mars5, testData4)

#create confusionMatrix
confusionMatrix(reference = testData$class, data = predicted_rf5, mode = 'everything')
