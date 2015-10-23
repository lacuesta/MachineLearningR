#setwd("/Users/Vicente/Documents/R/MachineLearningR/Kaggle/Titanic")
# Load datasets
titanic_train <- read.csv("./input/train.csv",stringsAsFactors = FALSE)
titanic_test <- read.csv("./input/test.csv",stringsAsFactors = FALSE)
summary(titanic_test)
summary(titanic_train)
# merge both datasets for imputation of missing values
titanic_test$Survived <- NA
titanic_all <- rbind(titanic_test, titanic_train)

# Convert to factors
titanic_all$Pclass <- factor(titanic_all$Pclass)
titanic_all$Embarked <- factor(titanic_all$Embarked, 
                               levels = c("S","C","Q"),
                               labels = c("Southampton", "Queenstown","Cherbourg"))
titanic_all$Sex <- factor(titanic_all$Sex)
titanic_all$Survived <- factor(titanic_all$Survived, levels = c(0,1),labels = c("no","yes"))

summary(titanic_all)
# We have the missing values:
# Age: 263 NAs
# Fare: 1 NA
# Embarked: 2

library(mice)
# impute missing values with mice() function
selected_cols <- c("Pclass","Sex","Age","SibSp", "Parch", "Fare", "Embarked")
imputation <- complete(mice(titanic_all[selected_cols]))
titanic_all[selected_cols] <- imputation

# split in train and test datasets again
titanic_train <- titanic_all[!is.na(titanic_all$Survived),]
titanic_test <- titanic_all[is.na(titanic_all$Survived),]
# delete Survived row in train dataset
titanic_test$Survived <- NULL

# Fit random forest model over different tunning parameters
library(caret)
titanic_rm <- train(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, data = titanic_train, method = "rf")
# It selects randomforest with mtry = 2
titanic_rm_pred <- predict(titanic_rm, titanic_test)

library(randomForest)
titanic_rm2 <- randomForest(Survived ~ Pclass + Sex + Age + SibSp + Parch + Fare + Embarked, 
                            data = titanic_train, mtry = 2)
titanic_rm_pred2 <- predict(titanic_rm2, titanic_test) 
titanic_test$Survived <- titanic_rm_pred2
# Convert to 0's and 1's
titanic_test$Survived <- as.numeric(titanic_test$Survived) - 1
write.csv(titanic_test[c("PassengerId","Survived")], "test_predictions.csv", row.names = FALSE)
