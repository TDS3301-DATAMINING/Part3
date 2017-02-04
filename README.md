# Part3

You can access the dataset at this [link](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+).

## A. Exploratory data analysis

The dataset choosen for this classification task is the occupancy detection data set. There is a total number of 7 attributes in the dataset which is temperature, relative humidity, light , CO2 , humidity ratio and occupancy. The dataset obtained from the source website is in three txt files where each of the files serves as purposes like training of the model, and the test data to test the model to see whether the model can classify the data correctly. The key attribute is temperature, light and CO2 for this task.

![solarized symmetry](https://github.com/TDS3301-DATAMINING/Part3/blob/master/graphs/outliers.PNG)

As plotted in the graph above, it is shown that there is some heavy outliers in both Light and CO2 attibutes. Besides that, there is also slight outliers found in humidity ratio and occupancy. 

## B. Pre-Processing tasks

* There is no missing value in all the datasets, so no imputation is required. 
```R
# load data
train <- read.table("occupancy_data/datatraining.txt", sep=",", header=T)
test <- read.table("occupancy_data/datatest.txt", sep=",", header=T)
test2 <- read.table("occupancy_data/datatest2.txt", sep=",", header=T)

# check if there are any missing values
sum(is.na(train)) # 0
sum(is.na(test)) # 0
sum(is.na(test2)) # 0
```

* We created a function to discretize the $date column.
```R
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

# relevel $WeekStatus into 0=weekend 1=weekday
train$WeekStatus <-unlist(lapply(train$WeekStatus, relevelWeekStatus))
test$WeekStatus <-unlist(lapply(test$WeekStatus, relevelWeekStatus))
test2$WeekStatus <-unlist(lapply(test2$WeekStatus, relevelWeekStatus))
```

* Then normalize the train data & test data for neural network.
```R
# scale data for nueral network
train.maxs <- apply(train[,2:8], 2, max)
train.mins <- apply(train[,2:8], 2, min)
train.scaled <- as.data.frame(scale(train[,2:8], center = train.mins, scale = train.maxs - train.mins))

test.maxs <- apply(train[,2:8], 2, max)
test.mins <- apply(train[,2:8], 2, min)
test.scaled <- as.data.frame(scale(train[,2:8], center = train.mins, scale = train.maxs - train.mins))

train_nn <- train.scaled
test_nn <- test.scaled
train_nn$Occupancy <- as.numeric(train_nn$Occupancy)
test_nn$Occupancy <- as.numeric(test_nn$Occupancy)
```

## C. Choice of performance measures

In this classification task, we are using ROC to measure each and every performance of the model used to perform classification on the dataset, which is decision tree, naive bayes and artificial neural network. In order to construct an ROC curve 4 values which are true positive, false positive, true negative and false negative is calculated. After that, those value are used to calculate the true positive rate and false positive rate which finally leads to plotting the ROC curve itself.

![solarized symmetry](https://github.com/TDS3301-DATAMINING/Part3/blob/master/graphs/roc.png)


## D. Performance of the 3 classifiers

* The accuracy of 3 classifiers are calculated as below:

```R
1. Decision Tree
    # create the tree
    library(tree)
    train.tree <- tree::tree(Occupancy ~ . -date, train)

    # do prediction
    train.tree.prediction <- predict(train.tree, test, type="class")
    
    # get the confusion matrix
    train.tree.confusion <- table(train.tree.pred, test$Occupancy)
    train.tree.confusion
         0    1
    0 1639    3
    1   54  969
     
    # compute the accuracy
    train.tree.accuracy <- sum(diag(train.tree.confusion)) / sum(train.tree.confusion)
    train.tree.accuracy # 0.9786116

2. Naive Bayes
    # create formula using the predictors name
    feats <- names(train.predictors)
    f <- paste(feats,collapse=' + ')
    f <- paste('Occupancy ~',f)
    f <- as.formula(f)

    # train the classifier
    library(e1071)
    train.naive <- naiveBayes(f, train)
    
    # do prediction
    train.naive.prediction <- predict(train.naive, test)
    
    # get the confusion matrix
    train.naive.confusion <- table(train.naive.prediction, test$Occupancy)
    train.naive.confusion
         0    1
    0 1638    5
    1   55  967
    
    # compute the accuracy
    train.naive.accuracy <- sum(diag(train.naive.confusion)) / sum(train.naive.confusion)
    train.naive.accuracy # 0.9774859 

3. Neural Networks
    # create formula from the header of train data
    feats <- names(train_nn[,c(-7)])
    feats
    f <- paste(feats,collapse=' + ')
    f <- paste('Occupancy ~',f)
    f <- as.formula(f)
    f

    # train the classifier with 1st and 2nd layers all 6 nodes
    library(neuralnet)
    train.nn <- neuralnet(f, train_nn, hidden=c(6,6), linear.output=FALSE) 
    train.nn
    
    # do prediction
    train.nn.predict <- compute(train.nn,test_nn[,c(-7)])
    train.nn.predict$net.result <- sapply(train.nn.predict$net.result,
                                             round,
                                             digits=0)
    
    # get the confusion matrix
    train.nn.confusion <- table(test_nn$Occupancy,train.nn.predict$net.result)
    train.nn.confusion
         0    1
    0 1638   55
    1    1  971
                                             
    # compute the accuracy
    train.nn.accuracy <- sum(diag(train.nn.confusion)) / sum(train.nn.confusion)
    train.nn.accuracy # 0.9789868668
```

* ROC graph is plotted to compare the three different classifiers. TPR & FPR are used as y-axis and x-axis. This is illustrated in Part C (above). 
```R
# Plot ROC graph with 3 curves representing decision tree, naive bayes, and neural networks

library(ROCR)
predict1 <- as.numeric(as.character(train.tree.prediction))
predict2 <- as.numeric(as.character(train.naive.prediction))
predict3 <- as.vector(train.nn.predict$net.result)

performance.tree <- performance(prediction(predict1, test$Occupancy), "tpr", "fpr" )
performance.naive <- performance(prediction(predict2, test$Occupancy), "tpr", "fpr" )
performance.nn <- performance(prediction(predict3, test_nn$Occupancy), "tpr", "fpr" )

plot(performance.tree, col="red", colorize=F, main="ROC")
plot(performance.naive, add=T, col="green")
plot(performance.nn, add=T, col = "blue")  

lines(performance.tree@x.values[[1]], performance.tree@y.values[[1]], col="red")
lines(performance.naive@x.values[[1]], performance.naive@y.values[[1]], col="green")
lines(performance.nn@x.values[[1]], performance.nn@y.values[[1]], col="blue")

legend('right', c("Decision Tree", "Naive Bayes", "Neural Networks"), lty=1,
     lwd=2, col=c("red", "green", "blue"), bty="n")
```

* AUC is calculated and can be observed from the ROC graph.
```R
# Decision Tree
auc(test$Occupancy, predict1) # auc.tree = 0.9825088

# Naive Bayes
auc(test$Occupancy, predict2) # auc.naive = 0.9811846

# Neural Networks
auc(test_nn$Occupancy, predict3) # auc.nn = 0.9840015
```

## E. Suggestion as to why the classifiers behave differently 

* Decision Tree
```
1. Decision tree is similar to human decision process.
2. Decision tree deal with both discrete and continuous features.
```

* Naive Bayes
```
1. Naive bayes takes independent assumptions between the features.
2. Naive bayes support missing values.
```

* Neural Networks
```
1. Neural network takes only numeric input.
2. Backpropagation can be used to update each of the weights in the network to minimize the error.
```

** Conclusion **

All the three classifiers scored an accuracy of > 90% and very close to each others.  
Decision tree score the highest accuracy among the tree, follow by neural networks and naive bayes.  
Weight settings and number of nodes in the hidden layer of neural networks is the thing to take note as it will affect the result of the accuracy. The activation function of a node also defines the output of that node given an input or set of inputs.
