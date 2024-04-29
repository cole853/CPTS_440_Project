from RF_Wrapper import *
from KNN_Wrapper import *
from itertools import combinations
import time


# this function checks to see if the accuracy is higher in the next RF or KNN.
# If the accuracy is higher, an updated tuple is returned. Otherwise, the input tuple is returned.
# Info is a tuple of (highest accuracy score, combination number for highest accuracy score)
# accuracy is the accuracy score to be checked
# number is the number of the predictors set associated with the accuracy
def check(info, accuracy, number):
    if len(info) == 0 or accuracy > info[0]:
        return tuple((accuracy, number))
    else:
        return info


# list of optional predictors:
# verified
# followers_count
# friends_count
# listed_count
# favourites_count
# statuses_count
# number_of_quoted_statuses
# number_of_retweeted_statuses
# total_retweet_count
# total_favorite_count
# total_hashtag_count
# total_url_count
# total_mentions_count
# total_media_count
# average_tweet_length
# average_retweet_count
# average_favorite_count
# average_hashtag_count
# average_url_count
# average_url_count
# average_mentions_count
# average_media_count
# random_numbers            (this is a control predictor for comparison)

predictor_list = [
    "verified",
    "followers_count",
    "friends_count",
    "listed_count",
    "favourites_count",
    "statuses_count",
    "number_of_quoted_statuses",
    "number_of_retweeted_statuses",
    "total_retweet_count",
    "total_favorite_count",
    "total_hashtag_count",
    "total_url_count",
    "total_mentions_count",
    "total_media_count",
    "average_tweet_length",
    "average_retweet_count",
    "average_favorite_count",
    "average_hashtag_count",
    "average_url_count",
    "average_url_count",
    "average_mentions_count",
    "average_media_count"
]

# get all combinations of 5 predictors
combinations_4 = combinations(predictor_list, 4)

# convert each combination to a list and add it to a list named combinations_4_list
combinations_4_list = []
for combination in combinations_4:
    combinations_4_list.append(list(combination))

print(len(combinations_4_list))

# takes 20 combinations to test changes to code
# combinations_4_list = combinations_4_list[:20]

# initialize combination_num
combination_num = 0

results_df = pd.DataFrame()

start_time = time.time()
for combination in combinations_4_list:

    # create random forest and K nearest neighbor
    test_KNN = KNN_Wrapper(predictors=combination, name=combination_num, n_neighbors=10, weights='uniform')
    print("combination " + str(combination_num + 1) + " / " + str(len(combinations_4_list)))
    test_RF = RF_Wrapper(predictors=combination, name=combination_num)

    # create new pandas dataframe to concat with the results dataframe
    temp_df = pd.DataFrame({
        "record_number"
        "IE_RF": [test_RF.IE_accuracy],
        "NS_RF": [test_RF.NS_accuracy],
        "TF_RF": [test_RF.TF_accuracy],
        "JP_RF": [test_RF.JP_accuracy],
        "IE_KNN": [test_KNN.IE_accuracy],
        "NS_KNN": [test_KNN.NS_accuracy],
        "TF_KNN": [test_KNN.TF_accuracy],
        "JP_KNN": [test_KNN.JP_accuracy]
        })

    # add the new row to the dataframe
    results_df = pd.concat([results_df, temp_df], ignore_index=True)
    results_df.reset_index()

    # add one to combination_num to differentiate name
    combination_num += 1

end_time = time.time()

# create two csv files with the information from results dataframe
results_df.to_csv("all_results.csv")
results_df.to_csv("backup.csv")
display(results_df)
print("This took " + str(round(end_time - start_time, 2)) + " seconds!")
