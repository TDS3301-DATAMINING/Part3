### A. EXPLORATORY DATA ANALYSIS

  # set working directory
  setwd("/Users/yongching/Desktop/Part3")
  
  # load data
  train <- read.table("occupancy_data/datatraining.txt", sep=",", header=T)
  test <- read.table("occupancy_data/datatest.txt", sep=",", header=T)
  test2 <- read.table("occupancy_data/datatest2.txt", sep=",", header=T)
  
  # reviewing the structure of the training set and testing set
  str(train)
  str(test)
  str(test2)
  
  # check for missing value
  sum(is.na(train)) # 0
  sum(is.na(test)) # 0
  sum(is.na(test2)) # 0
  
  # check for outliers
  boxplot(train)

### B. PRE-PROCESSING TASKS
  
  # load library
  library(lubridate)
  
  checkWeekStatus <- function(x) {
    val <- weekdays(x)
    if (val == "Saturday" | val == "Sunday") {
      val2 = "Weekend"
    }
    else {
      val2 = "Weekday"
    }
    return(val2)
  }
  
  relevelWeekStatus <- function(x) {
    if (x == "Weekend") {
      val2 = 0
    }
    else {
      val2 = 1
    }
    return(val2)
  }
  
  # convert $date column as POSIXct format
  train$date <- as.POSIXct(train$date,tz="UTC") 
  test$date <- as.POSIXct(test$date,tz="UTC") 
  test2$date <- as.POSIXct(test2$date,tz="UTC") 
  
  # discretize the $date column
  train$WeekStatus <-unlist(lapply(train$date, checkWeekStatus))
  test$WeekStatus <-unlist(lapply(test$date, checkWeekStatus))
  test2$WeekStatus <-unlist(lapply(test2$date, checkWeekStatus))
  
  # relevel $WeekStatus to 0=weekend 1=weekday
  train$WeekStatus <-unlist(lapply(train$WeekStatus, relevelWeekStatus))
  test$WeekStatus <-unlist(lapply(test$WeekStatus, relevelWeekStatus))
  test2$WeekStatus <-unlist(lapply(test2$WeekStatus, relevelWeekStatus))
  
  # for nueral network
  train_nn <- train[,2:8]
  test_nn <- test[,2:8]
  train_nn$Occupancy <- as.numeric(train_nn$Occupancy)
  test_nn$Occupancy <- as.numeric(test_nn$Occupancy)
  
  # convert $WeekStatus column as factor
  train$WeekStatus <- as.factor(train$WeekStatus)
  test$WeekStatus <- as.factor(test$WeekStatus)
  test2$WeekStatus <- as.factor(test2$WeekStatus)
  
  # convert $occupancy column as factor
  train$Occupancy <- as.factor(train$Occupancy)
  test$Occupancy <- as.factor(test$Occupancy)
  test2$Occupancy <- as.factor(test2$Occupancy)
  
  # reorder the columns
  train_nn <- train_nn[ , c(7,1:6)]
  test_nn <- test_nn[ , c(7,1:6)]
  train <- train[ , c(1,8,2:7)]
  test <- test[ , c(1,8,2:7)]
  test2 <- test2[ , c(1,8,2:7)]
  
  # review the structure again
  str(train_nn)
  str(test_nn)
  str(train)
  str(test)
  str(test2)

### D. PERFORMANCE OF 3 CLASSIFIER
  
  # ~~~ 1. Decision tree ~~~
  library(tree)
  train.tree <- tree::tree(Occupancy ~ . -date, train)
    
    # list the rules
    train.tree
    
    # summary of the tree
    summary(train.tree)
    summary(train.tree)$misclass # 99/8143 misclassification rate which is pretty good
    
    # check what are the variables used by the tree
    summary(train.tree)$used
    
    # plot the tree
    plot(train.tree)
    text(train.tree,pretty=0)
    
    # test the tree
    train.tree.pred <- predict(train.tree, test, type="class")
    
    # confusion matrix
    train.tree.confusion <- table(train.tree.pred, test$Occupancy)
    train.tree.confusion
    
    # compute the accuracy, TPR and FPR
    train.tree.accuracy <- sum(diag(train.tree.confusion)) / sum(train.tree.confusion)
    train.tree.accuracy # 0.9786116
    
    
  # ~~~ 2. Naive bayes ~~~
    
    # Feature selection
    
      # use seed to ensure results are repeatable
      set.seed(123)
      
      # load libraries
      library(mlbench)
      library(lattice)
      library(ggplot2)
      library(caret)
      library(randomForest)
      
      # define the control using a random forest selection function
      control <- rfeControl(functions=rfFuncs, method="cv", number=10)
      
      # run Recursive Feature Elimination(RFE) algorithm with different subsets
      # from the training data and identify the features required to build an accurate model
      results <- rfe(train[,2:7], train[,8], sizes=c(2:7), rfeControl=control)
      
      # summarize the results
      print(results)
      
      # list the chosen features and store into a variable
      predictors(results) # Light, CO2, Temperature
      train.predictors <- train[predictors(results)]
      
      # plot and save the results
      plot(results, type=c("g", "o"))
    
    # Execution
      
      # load library
      library(e1071)
      
      # create formula using the predictors name
      feats <- names(train.predictors)
      f <- paste(feats,collapse=' + ')
      f <- paste('Occupancy ~',f)
      f <- as.formula(f)
      
      # train the classifier
      train.naive <- naiveBayes(f, train)
      
      # test it 
      train.naive.prediction <- predict(train.naive, test)
      
      # get the confusion matrix
      train.naive.confusion <- table(train.naive.prediction, test$Occupancy)
      train.naive.confusion
    
      # compute the accuracy
      train.naive.accuracy <- sum(diag(train.naive.confusion)) / sum(train.naive.confusion)
      train.naive.accuracy # 0.9774859  
      
  # ~~~ 3. Artificial nueral networks ~~~
    
    # load library
    library(neuralnet)
    
    # convert to formula
    feats <- names(train_nn[,c(-7)])
    feats
    f <- paste(feats,collapse=' + ')
    f <- paste('Occupancy ~',f)
    f <- as.formula(f)
    f
    
    #1st and 2nd layers all 5 nodes
    train.nn <- neuralnet(f, train_nn, hidden=c(6,6), linear.output=FALSE) 
    train.nn
    
    # Compute Predictions off test set
    train.nn.predict <- compute(train.nn,test_nn[,c(-7)])
    
    # Print the result
    print(train.nn.predict)
    print(train.nn.predict$net.result)
    
    # round to 0 or 1, because we want to compare it
    train.nn.predict$net.result <- sapply(train.nn.predict$net.result,
                                             round,
                                             digits=0)
    
    # confusion matrix
    train.nn.confusion <- table(test_nn$Occupancy,train.nn.predict$net.result)
    train.nn.confusion
    
    # compute the accuracy, TPR and FPR
    train.nn.accuracy <- sum(diag(train.nn.confusion)) / sum(train.nn.confusion)
    train.nn.accuracy # 0.9789868668
    
    # plot ann
    plot(train.nn)
    
    # plot ROC and compute the AUC
    library(ROCR)
    predict <- as.vector(train.nn.predict$net.result)
    pred <- prediction(predict,test_nn$Occupancy)
    roc = performance(pred, "tpr", "fpr")
    plot(roc, lwd=2, colorize=TRUE)
    lines(x=c(0, 1), y=c(0, 1), col="black", lwd=1)
    auc = performance(pred, "auc")
    auc = unlist(auc@y.values)
    auc # 0.9840015411
    