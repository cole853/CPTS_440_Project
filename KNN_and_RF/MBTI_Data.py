from IPython.display import display
import pandas as pd
import numpy as np

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', None)

class MBTI_Data:

    def __init__(self, testPercent=0.8):
        # get mbti labels from csv file and make a pandas dataframe
        unsorted_mbti_df = pd.read_csv('mbti_labels.csv')

        # get user info from csv file and make a pandas dataframe
        unsorted_user_info_df = pd.read_csv('user_info.csv')

        # sort both dataframes so users are at the same position as their results
        self.user_data = unsorted_user_info_df.sort_values(by='id')
        self.mbti = unsorted_mbti_df.sort_values(by='id')

        # reset the row index after sort
        self.user_data.reset_index(drop=True, inplace=True)
        self.mbti.reset_index(drop=True, inplace=True)

        # remove unnecessary columns from user info
        drop_columns = ['id_str', 'name', 'screen_name', 'location', 'description', 'number_of_tweets_scraped']
        self.user_data = self.user_data.drop(columns=drop_columns)

        # change bool verified column to int
        self.user_data['verified'] = self.user_data['verified'].astype(int)

        # create a column of random values as a control predictor to compare real predictors to.
        total_rows = self.user_data.shape[0]
        self.user_data['random_numbers'] = np.random.rand(total_rows)

        # add a column for each letter of the personality type. each letter is represented by 1 or 0. An intj
        # personality will be represented by 1 for each letter column while an esfp personality will be represented
        # by 0 in each letter column.
        for letter in 'intj':
            self.mbti[letter] = self.mbti['mbti_personality'].str.contains(letter).astype(int)

        # split the dataframes into train and test sets
        self.train_user_data = pd.DataFrame()
        self.train_mbti = pd.DataFrame()
        self.test_user_data = pd.DataFrame()
        self.test_mbti = pd.DataFrame()
        self.splitTrainTest(testPercent)

    # splits the dataframes into train and test sets based on the percent entered.
    # for example entering 0.8 will make training sets containing the first 80% of the total records and test sets
    # containing the remaining 20%
    def splitTrainTest(self, testPercent):
        self.train_user_data = self.user_data.iloc[:(int(len(self.user_data) * testPercent))]
        self.train_mbti = self.mbti.iloc[:(int(len(self.mbti) * testPercent))]

        self.test_user_data = self.user_data.iloc[(int(len(self.user_data) * testPercent)):]
        self.test_mbti = self.mbti.iloc[(int(len(self.mbti) * testPercent)):]

    # show the dataframes
    def showDataframes(self):
        print("########################################### Data Frames ##############################################")

        # print all MBTI data
        print("Full MBTI Data")
        display(self.mbti)

        # print all User Data
        print("Full User Data")
        display(self.user_data)

        # print the MBTI training set
        print("Train MBTI Data")
        display(self.train_mbti)

        # print the User training set
        print("Train User Data")
        display(self.train_user_data)

        # print the MBTI test set
        print("Test MBTI Data")
        display(self.test_mbti)

        # print the User test set
        print("Test User Data")
        display(self.test_user_data)

        print("######################################### End Data Frames ############################################")
