from itertools import combinations
import NaiveBayes_Wrapper
from IPython.display import display
import pandas as pd
import time

# Also the same as cole's code, print report accurately and easily compared to other tests


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

# get all combinations of 4 predictors
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

    # create Naive Bayes model
    test_NB = NaiveBayes_Wrapper(predictors=combination, name=combination_num, confidence=0.5)
    print("combination " + str(combination_num + 1) + " / " + str(len(combinations_4_list)))

    # create new pandas dataframe to concat with the results dataframe
    temp_df = pd.DataFrame({
        "record_number": combination_num,
        "IE_NB": [test_NB.IE_accuracy],
        "NS_NB": [test_NB.NS_accuracy],
        "TF_NB": [test_NB.TF_accuracy],
        "JP_NB": [test_NB.JP_accuracy]
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