#install library
class_pack = c("caret", "skimr", "RANN","randomForest","fastAdaboost","gbm","xgboost", "caretEnsemble","C50","earth")
install.packages(class_pack)

#include library
library(caret)
library(skimr)
library(RANN)

#Load url dataset
orange <- read.csv('https://raw.githubusercontent.com/selva86/datasets/master/orange_juice_withmissing.csv')

#divide training testing
str(orange)
trainRowNumbers = createDataPartition(orange$Purchase, p = 0.8, list = FALSE)

#create train, test dataset
trainData = orange[trainRowNumbers,]
testData = orange[-trainRowNumbers,]

# x coordinate / y coordinate
x = trainData[,1:20]
y = trainData$Purchase

skimmed = skim_to_wide(trainData)
skimmed[,c(10:16)]

#check N/A
anyNA(trainData)
anyNA(testData)

#hot encording
preProcess_missingdata_model = preProcess(trainData, method = 'knnImpute')
trainData_impute = predict(preProcess_missingdata_model,newdata = trainData)
dummies_model = dummyVars(Purchase ~., data = trainData_impute)
trainData_mat = predict(dummies_model, newdata = trainData_impute)
trainData_dummy = data.frame(trainData_mat)

#range
preProcess_range_model = preProcess(trainData_dummy, method = 'range')
trainData_pre = predict(preProcess_range_model, newdata = trainData_dummy)
apply(trainData_pre[, 1:10], 2, FUN=function(x){c('min'=min(x), 'max'=max(x))})
trainData_pre$Purchase = y 

#traindata trainning
model_rf = train(Purchase ~., data = trainData_pre, method = 'rf')
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
fitControl <- trainControl( method = 'cv', number = 5, savePredictions = 'final', classProbs = T, summaryFunction = twoClassSummary)

# training earth/rf/svmRadial/nnet
model_mars2 = train(Purchase ~., data = trainData_pre, method = 'earth', metric = 'ROC', tuneLength = 5, trControl = fitControl)
model_mars3 = train(Purchase ~., data = trainData_pre, method = 'rf', metric = 'ROC', tuneLength = 5, trControl = fitControl)
model_mars4 = train(Purchase ~., data = trainData_pre, method = 'svmRadial', metric = 'ROC', tuneLength = 5, trControl = fitControl)
model_mars5 = train(Purchase ~., data = trainData_pre, method = 'nnet', metric = 'ROC', tuneLength = 5, trControl = fitControl)

#predict
predicted_rf2 = predict(model_mars2, testData4)
predicted_rf3 = predict(model_mars3, testData4)
predicted_rf4 = predict(model_mars4, testData4)
predicted_rf5 = predict(model_mars5, testData4)

#create confusionMatrix
confusionMatrix(reference = testData$Purchase, data = predicted_rf2, mode = 'everything', positive = 'MM')
confusionMatrix(reference = testData$Purchase, data = predicted_rf3, mode = 'everything', positive = 'MM')
confusionMatrix(reference = testData$Purchase, data = predicted_rf4, mode = 'everything', positive = 'MM')
confusionMatrix(reference = testData$Purchase, data = predicted_rf5, mode = 'everything', positive = 'MM')
