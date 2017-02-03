# Part3

You can access the dataset at this [link](https://archive.ics.uci.edu/ml/datasets/Occupancy+Detection+).

## A. Exploratory data analysis

The dataset choosen for this classification task is the occupancy detection data set. There is a total number of 7 attributes in the dataset which is temperature, relative humidity, light , CO2 , humidity ratio and occupancy. The dataset obtained from the source website is in three txt files where each of the files serves as purposes like training of the model, and the test data to test the model to see whether the model can classify the data correctly. The key attribute is temperature, light and CO2 for this task.

![solarized symmetry](https://github.com/TDS3301-DATAMINING/Part3/blob/master/graphs/outliers.PNG)

As plotted in the graph above, it is shown that there is some heavy outliers in both Light and CO2 attibutes. Besides that, there is also slight outliers found in humidity ratio and occupancy. 

## C. Choice of performance measures

In this classification task, we are using ROC to measure each and every performance of the model used to perform classification on the dataset, which is decision tree, naive bayes and artificial neural network. In order to construct an ROC curve 4 values which are true positive, false positive, true negative and false negative is calculated. After that, those value are used to calculate the true positive rate and false positive rate which finally leads to plotting the ROC curve itself.

![solarized symmetry](https://github.com/TDS3301-DATAMINING/Part3/blob/master/graphs/roc.png)


