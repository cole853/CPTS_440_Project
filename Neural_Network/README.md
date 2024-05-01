# Neural Network Documentation

**libraries used:**

- IPython

- pandas

- numpy

- sklearn

- time

- os

- itertools

**Files used:**

- mbti_labels.csv this file contains the meyers briggs personality score for each user and their ID number

- user_info.csv this file contains the user info and ID number for each user.

- Neural_Network: this is a jupyter notebook file that contains everything used to run the program make sure to download both csv files in order to run it.

# MBTI_Data documentation
- This class genrally reads MBTI labels and user info from the CSV files and sorts both dataframes based on their ID. It then drops any unnecessary columns from the user dataframe, it then convert the bool "verfied" 
column to an int type. Adds columns for each letter of the personality type, representing each letter either 1 or 0 then splits the data into training and test data which is 80/20.

splitTrainTest()

- Splits the dataframes into train and test sets based on the provided percentage.

showDataFrames()

- Displays various dataframes including full MBTI data, full user data, train MBTI data, train user data, test MBTI data, and test user data.

Instantiation and Diplay
- Instantiatates the MBTI class and class showDataFrames() method to display the data frame

# MLP_Wrapper documentation

Constructor

- Initializes the MBTI_Data object to handle data preprocessing and sets various parameters for the MLP classifiers. Instantiate the four MLP classifiers for each personality letter position: IE, NS, TF, and JP.
Then make predictions on the test data, evaluates performance, prints results, and writes a full report.

fitClassifiers()

- Fits each MLP classifier to its corresponding personality letter position using the training data.

makePredictionsAll()

- Makes predictions for each personality letter position using the test data and combines predictions with actual labels into dataframes, we then calculate correctness and accuracy for each personality letter position.

makePredictions()

- Makes predictions using a specified MLP classifier and returns the predicted probabilities.

setResults()

- Calculates correctness and accuracy for each personality letter position based on the combined dataframes of predictions and actual labels.

printPredictions()

- Prints a dataframe for each letter position that includes predictions, target values, and right/wrong column.

writeReport()

- Writes a report to the a text file which includes model setting and evaluation results


