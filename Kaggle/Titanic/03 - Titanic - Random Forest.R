#setwd("/Users/Vicente/Documents/R/MachineLearningR/Kaggle/Titanic")

# Same as 01 - Titanic - Random Forest but using more features
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
selected_cols <- c("Pclass","Sex","Age","SibSp", "Parch", "Fare", "Embarked")
imputation <- complete(mice(titanic_all[selected_cols]))
titanic_all[selected_cols] <- imputation

# split in train and test datasets again
titanic_train <- titanic_all[!is.na(titanic_all$Survived),]
titanic_test <- titanic_all[is.na(titanic_all$Survived),]
# delete Survived row in train dataset
titanic_test$Survived <- NULL

predictors <- c("Pclass","Sex","Age","SibSp","Parch","Fare","Embarked","Title")
response <- c("Survived")

titanic_train_dummy <- dummyVars(~ Pclass + Embarked + Title, titanic_train)
titanic_train_dummy_df <- as.data.frame(predict(titanic_train_dummy, titanic_train))
titanic_train_dummy_df <- data.frame(titanic_train[c("Sex","Age","SibSp","Parch","Fare")],
                                     titanic_train_dummy_df, titanic_train["Survived"] )
# titanic_rf <- train(Survived ~ ., data = titanic_train_dummy_df)
# the optimal is with mtry = 14, acuracy = 0.8112


titanic_rf2 <- randomForest(Survived ~ ., 
                           data = titanic_train_dummy_df, mtry = 14)

titanic_test_dummy <- dummyVars(~ Pclass + Embarked + Title, titanic_test)
titanic_test_dummy_df <- as.data.frame(predict(titanic_test_dummy, titanic_test))
titanic_test_dummy_df <- data.frame(titanic_test[c("Sex","Age","SibSp","Parch","Fare")],
                                     titanic_test_dummy_df)

titanic_rf2_pred <- predict(titanic_rf2, titanic_test_dummy_df) 
titanic_test$Survived <- titanic_rf2_pred
# Convert to 0's and 1's
titanic_test$Survived <- as.numeric(titanic_test$Survived) - 1
write.csv(titanic_test[c("PassengerId","Survived")], "test_predictions.csv", row.names = FALSE)
