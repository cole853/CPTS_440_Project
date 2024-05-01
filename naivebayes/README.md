# Naive Bayes Documentation
# For reference, this program requires Python 3.9.12 specifically. It's able to run in the google collab version without needing to potentially downgrade, but the csv's will still need to be downloaded for temporary use.

**libraries used:**

- IPython

- pandas

- numpy

- sklearn.naive_bayes (GaussianNB)

- time

- os

- itertools

**Files used:**

- mbti_labels.csv this file contains the meyers briggs personality score for each user and their ID number

- user_info.csv this file contains the user info and ID number for each user.

- NaiveBayes_Wrapper.py is the wrapper class for Naive Bayes.

- MBTI_Data.py contains the MBTI_Data class

- main.py contains the instances of RF_Wrapper and KNN_Wrapper. This is where the program is run.

# MBTI_Data documentation

Constructor

- The constructor for MBTI_Data class reads data from mbti_labels.csv and user_info.csv. It sorts both by id number so the mbti is matched with user data. Then unnessesary columns are dropped and a random numbers column is created to compare with the results of real predictors. New columns are created with i, n, t, and j labels. These values are 1 if the corresponding letter is part of the personality otherwise 0. each letter position can only be one of two options which makes 0 and 1 a good way to represent it. The dataframes are split into train and test sets.

splitTrainTest()

- This function divides the dataframes into train and test sets. This is called in the constructor.

showDataFrames()

- This function prints the full mbti dataframe, full user_data dataframe, and then train and test sets for both.

# RF_Wrapper documentation

Constructor

- Initializes result values and random forests. Each random forest is then fit to the train data, predictions are made, results are printed, and a new file is created to record the results.

fitClassifiers()

- Fits each classifier to its corresponding personality letter position. This is called in the constructor.

makePredictionsAll()

- Makes predictions for each letter position using the makePredictions() function and creates pandas dataframes that have a prediction column and target column. A right/wrong column is also added to these dataframes which is true if the target and prediction match, false otherwise. After this, the results are set using the setResults() function.

makePredictions()

- Makes and returns the predictions for one letter position. called in makePredictionsAll() function.

setResults()

- Sets all the result values for each letter position. This includes the total number of predictions, total number of correct predictons, and accuracy percentage (correct / total). This function is called in makePredictionsAll().

printData()

- Calls the showDataframes() function in MBTI_Data class to show both full dataframes along with train and test sets.

printPredictions()

- Prints a dataframe for each letter position that includes predictions, target values, and right/wrong column.

writeReport()

- Writes a string to a result file. This function is called in writeFullReport().

writeFullReport()

- Writes settings of classifiers and the results to a file using the writeReport() function. In the same format as printResults().

printResults()

- Prints settings of classifiers and the results to the tereminal. In the same format as writeFullReport().