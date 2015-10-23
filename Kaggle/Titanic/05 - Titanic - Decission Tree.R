#setwd("/Users/Vicente/Documents/R/MachineLearningR/Kaggle/Titanic")

# Same as 02 - Titanic - Try to use Decision tree
# Load datasets
titanic_train <- read.csv("./input/train.csv",stringsAsFactors = FALSE)
titanic_test <- read.csv("./input/test.csv",stringsAsFactors = FALSE)
#summary(titanic_test)
#summary(titanic_train)
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

# Extract titles
extractTitle <- function (x){
  title = regmatches(x,regexec("\\w+\\.",x))
  title = gsub("[.]","",unlist(title))
  return(unlist(title))
}
titanic_all$Title <- factor(unlist(lapply(titanic_all$Name,extractTitle)))

summary(titanic_all)
# We have the missing values:
# Age: 263 NAs
# Fare: 1 NA
# Embarked: 2

library(mice)
# impute missing values with mice() function
selected_cols <- c("Pclass","Sex","Age","SibSp", "Parch", "Fare", "Embarked","Title")
imputation <- complete(mice(titanic_all[selected_cols]))
titanic_all[selected_cols] <- imputation

# split in train and test datasets again
titanic_train <- titanic_all[!is.na(titanic_all$Survived),]
titanic_test <- titanic_all[is.na(titanic_all$Survived),]

# delete Survived row in train dataset
titanic_test$Survived <- NULL



predictors <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title")
response <- c("Survived")
library(C50)
#titanic_c50 <- train(Survived ~ ., data = titanic_train[c(predictors,response)], method = "C5.0")
#titanic_c50
# Best: trials = 20, model = tree, winnow = TRUE
titanic_c50 <- C5.0(titanic_train[predictors], titanic_train$Survived, trials = 20, tree = TRUE)
titanic_c50_pred <- predict(titanic_c50, titanic_test[predictors])
titanic_test$Survived <- titanic_c50_pred
# Convert to 0's and 1's
titanic_test$Survived <- as.numeric(titanic_test$Survived) - 1
write.csv(titanic_test[c("PassengerId","Survived")], "test_predictions_c50.csv", row.names = FALSE)
