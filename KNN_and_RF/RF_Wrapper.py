from sklearn.ensemble import RandomForestClassifier
from MBTI_Data import *
import time
import os


class RF_Wrapper:

    # initializes an instance of RF_Wrapper
    # predictors is a list of column names used to predict values
    # name is the name of the report file that will be generated
    # n_estimators is the number of trees to be created in the forest
    # min_samples_split is the minimum number of records needed to split a node
    # random_state controls the random number generation in the random forest making results reproducible when set.
    # confidence is the cutoff for a value to be classified as one
    #   if the random forest returns a value greater than or equal to the confidence it will be classified as one,
    #   0 otherwise
    def __init__(self, predictors, name="", n_estimators=100, min_samples_split=350, random_state=1, confidence=0.5):

        # Get the data for training and testing. Training data is the first 80% if no value is entered.
        # Training data percent can be changed by entering the new percent in MBTI constructor.
        # For example entering 0.5 would split training and testing sets evenly.
        self.data = MBTI_Data()

        # create a name for the report if none was given
        if name == "":
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            self.name = str(timestamp)
        else:
            self.name = name

        # set values
        self.predictors = predictors
        self.n_estimators = n_estimators
        self.min_samples_split = min_samples_split
        self.random_state = random_state
        self.confidence = confidence

        # initialize predictions to empty pandas series
        self.IE_combined = pd.DataFrame()
        self.NS_combined = pd.DataFrame()
        self.TF_combined = pd.DataFrame()
        self.JP_combined = pd.DataFrame()

        # initialize results
        self.IE_total = 0
        self.NS_total = 0
        self.TF_total = 0
        self.JP_total = 0

        self.IE_correct = 0
        self.NS_correct = 0
        self.TF_correct = 0
        self.JP_correct = 0

        self.IE_accuracy = 0.0
        self.NS_accuracy = 0.0
        self.TF_accuracy = 0.0
        self.JP_accuracy = 0.0

        # create classifier used to determine Introversion-Extroversion (I or E)
        self.IE = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # create classifier used to determine Intuition-Sensing (N or S)
        self.NS = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # create classifier used to determine Thinking-Feeling (T or F)
        self.TF = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # create classifier used to determine Judging-Perceiving (J or P)
        self.JP = RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_split=min_samples_split,
            random_state=random_state
        )

        # fit RandomForestClassifiers
        self.fitClassifiers()

        # make predictions
        self.makePredictionsAll()

        # print results and write report
        self.printResults()
        self.writeFullReport()

    # fit all classifiers using predictors to their corresponding value
    def fitClassifiers(self):
        self.IE.fit(self.data.train_user_data[self.predictors], self.data.train_mbti["i"])
        self.NS.fit(self.data.train_user_data[self.predictors], self.data.train_mbti["n"])
        self.TF.fit(self.data.train_user_data[self.predictors], self.data.train_mbti["t"])
        self.JP.fit(self.data.train_user_data[self.predictors], self.data.train_mbti["j"])

    # make predictions for each random forest using the self.makePredictions() function
    def makePredictionsAll(self):

        i = self.data.test_user_data.index

        # get predictions
        predictions_IE = pd.Series(self.makePredictions(self.IE), index=i, name="IE_Predictions")
        predictions_NS = pd.Series(self.makePredictions(self.NS), index=i, name="NS_Predictions")
        predictions_TF = pd.Series(self.makePredictions(self.TF), index=i, name="TF_Predictions")
        predictions_JP = pd.Series(self.makePredictions(self.JP), index=i, name="JP_Predictions")

        # create pandas dataframes that include predictions and target values
        self.IE_combined = pd.concat([self.data.test_mbti["i"], predictions_IE], axis=1)
        self.NS_combined = pd.concat([self.data.test_mbti["n"], predictions_NS], axis=1)
        self.TF_combined = pd.concat([self.data.test_mbti["t"], predictions_TF], axis=1)
        self.JP_combined = pd.concat([self.data.test_mbti["j"], predictions_JP], axis=1)

        # create a new column right/wrong that shows whether the prediction was right as a boolean
        self.IE_combined["right/wrong"] = (self.IE_combined['i'] == self.IE_combined['IE_Predictions'])
        self.NS_combined["right/wrong"] = (self.NS_combined['n'] == self.NS_combined['NS_Predictions'])
        self.TF_combined["right/wrong"] = (self.TF_combined['t'] == self.TF_combined['TF_Predictions'])
        self.JP_combined["right/wrong"] = (self.JP_combined['j'] == self.JP_combined['JP_Predictions'])

        self.setResults()

    # make predictions for one of the random forests
    # RF is the random forest that has already been fit to data
    # returns predictions for all test records
    def makePredictions(self, RF):
        predictions = RF.predict_proba(self.data.test_user_data[self.predictors])[:, 1]
        predictions[predictions >= self.confidence] = int(1)
        predictions[predictions < self.confidence] = int(0)
        return predictions

    # sets the results for each random forest after predictions have been made.
    # This function is called in the makePredictionsAll() function.
    def setResults(self):
        self.IE_total = self.IE_combined.shape[0]
        self.IE_correct = (self.IE_combined['right/wrong'] == True).sum()
        self.IE_accuracy = self.IE_correct / self.IE_total

        self.NS_total = self.NS_combined.shape[0]
        self.NS_correct = (self.NS_combined['right/wrong'] == True).sum()
        self.NS_accuracy = self.NS_correct / self.NS_total

        self.TF_total = self.TF_combined.shape[0]
        self.TF_correct = (self.TF_combined['right/wrong'] == True).sum()
        self.TF_accuracy = self.TF_correct / self.TF_total

        self.JP_total = self.JP_combined.shape[0]
        self.JP_correct = (self.JP_combined['right/wrong'] == True).sum()
        self.JP_accuracy = self.JP_correct / self.JP_total

    # prints the full MBTI data, full User data, train MBTI data, train user data, test MBTI data, and test user data
    def printData(self):
        self.data.showDataframes()

    # prints predictions with their target values
    def printPredictions(self):

        print("########################################### Predictions ##############################################")
        display(self.IE_combined)
        display(self.NS_combined)
        display(self.TF_combined)
        display(self.JP_combined)
        print("######################################### End Predictions ############################################")

    # writes the input string to a result file
    def writeReport(self, string, noNewline=False):
        os.makedirs("Reports", exist_ok=True)
        file_path = os.path.join("Reports", str(self.name) + "_RF.txt")

        with open(file_path, 'a') as file:
            if noNewline:
                file.write(string)
            else:
                file.write(str(string))
                file.write("\n")

    #writes a full report to result file calls writeReport() function
    def writeFullReport(self):

        self.writeReport("\nPredictors: ")
        self.writeReport(str(self.predictors))
        self.writeReport("\nn_estimators:\t\t\t" + str(self.n_estimators))
        self.writeReport("min_samples_split:\t\t" + str(self.min_samples_split))
        self.writeReport("random_state:\t\t\t" + str(self.random_state) + "\n")
        self.writeReport("Confidence:\t\t\t\t" + str(self.confidence))
        self.writeReport(
            "-----------------------------------------------------\n\n"
            "-----------------------------------------------------"
        )

        self.writeReport("IE Correct:\t\t\t\t" + str(self.IE_correct))
        self.writeReport("IE Total:\t\t\t\t" + str(self.IE_total))
        self.writeReport("IE Percent Correct:\t\t" + str(self.IE_accuracy) + "\n")

        self.writeReport("NS Correct:\t\t\t\t" + str(self.NS_correct))
        self.writeReport("NS Total:\t\t\t\t" + str(self.NS_total))
        self.writeReport("NS Percent Correct:\t\t" + str(self.NS_accuracy) + "\n")

        self.writeReport("TF Correct:\t\t\t\t" + str(self.TF_correct))
        self.writeReport("TF Total:\t\t\t\t" + str(self.TF_total))
        self.writeReport("TF Percent Correct:\t\t" + str(self.TF_accuracy) + "\n")

        self.writeReport("JP Correct:\t\t\t\t" + str(self.JP_correct))
        self.writeReport("JP Total:\t\t\t\t" + str(self.JP_total))
        self.writeReport("JP Percent Correct:\t\t" + str(self.JP_accuracy) + "\n")

    def printResults(self):

        print("############################################# Results ################################################")
        print("\nPredictors: ")
        print(str(self.predictors))
        print("\nn_estimators:\t\t\t" + str(self.n_estimators))
        print("min_samples_split:\t\t" + str(self.min_samples_split))
        print("random_state:\t\t\t" + str(self.random_state) + "\n")
        print("Confidence:\t\t\t\t" + str(self.confidence))
        print(
            "-----------------------------------------------------\n\n"
            "-----------------------------------------------------"
        )

        print("IE Correct:\t\t\t\t" + str(self.IE_correct))
        print("IE Total:\t\t\t\t" + str(self.IE_total))
        print("IE Percent Correct:\t\t" + str(self.IE_accuracy) + "\n")

        print("NS Correct:\t\t\t\t" + str(self.NS_correct))
        print("NS Total:\t\t\t\t" + str(self.NS_total))
        print("NS Percent Correct:\t\t" + str(self.NS_accuracy) + "\n")

        print("TF Correct:\t\t\t\t" + str(self.TF_correct))
        print("TF Total:\t\t\t\t" + str(self.TF_total))
        print("TF Percent Correct:\t\t" + str(self.TF_accuracy) + "\n")

        print("JP Correct:\t\t\t\t" + str(self.JP_correct))
        print("JP Total:\t\t\t\t" + str(self.JP_total))
        print("JP Percent Correct:\t\t" + str(self.JP_accuracy) + "\n")

        print("########################################### End Results ##############################################")
