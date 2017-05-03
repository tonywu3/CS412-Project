#Compatibility with py2
from __future__ import division
from builtins import *

import csv
import numpy as np


from collections import Counter

def get_gender(userID):
    return user_info_dict.get(userID)[0]

def calculate_deviations(total_array, counts_array, avg, size):
    deviation = [0 for i in range(size)]
    for i in range(size):
        try:
            curr_avg = total_array[i] / counts_array[i]
            deviation[i] = curr_avg - avg
        except:
            deviation[i] = 0.0
    return deviation


#def calculate_movie_deviations():
#    for i in range(movies):




def predict(test_array, user_deviation, movie_deviation):
    ratings = []
    for item in test_array:
        #print(item)
        curr_user = int(np.asscalar(np.asarray(item[1])))
        curr_movie = int(np.asscalar(np.asarray(item[2])))
        curr_gender = get_gender(curr_user)

        gender_offset = 0.0
        if (curr_gender == 'M'):
            gender_offset = male_gender_diffs[curr_movie]
        elif (curr_gender == 'F'):
            gender_offset = female_gender_diffs[curr_movie]

        curr_score = avg_rating + user_deviation[curr_user] + movie_deviation[curr_movie] #+ gender_offset * .2
        #print("user {}, movie {}, {} -> {}".format(item[1], item[2], curr_score, round(curr_score)))
        ratings.append(str(int(round(curr_score))))

    ratings = np.array(ratings)

    return ratings

#Beginning of code

#Parse movie info file
with open('movie.txt', 'r') as movie_CSV:
    movie_info_reader = csv.reader(movie_CSV)

    #First row contains header, no need to store it
    next(movie_info_reader)

    #Get the values of the movie information into a dictionary with the key being the movie ID
    #Also find unique categories
    movie_categories = set()
    movie_info_dict = {}
    for curr_row in movie_info_reader:
        movie_categories.update(curr_row[2].split("|"))
        movie_info_dict[curr_row[0]] = curr_row[1:]


#Make sure order of categories stays the same when used later
movie_categories_list = list(movie_categories)
movie_categories_list.remove('N/A')

#Prevent accidental usage of unordered set later on
del movie_categories

#Parse user info file
with open('user.txt', 'r') as user_CSV:
    user_info_reader = csv.reader(user_CSV)

    #First row contains header, no need to store it
    next(user_info_reader)

    #Get the values of the user information into a dictionary with the key being the user ID
    user_info_dict = {}
    for curr_row in user_info_reader:
        user_info_dict[int(curr_row[0])] = curr_row[1:]


#Parse training data file
with open('train.txt', 'r') as train_CSV:
    training_reader = csv.reader(train_CSV)

    #First row contains header, no need to store it
    next(training_reader)

    data_array = []

    for curr_row in training_reader:
        data_array.append(curr_row)
    data_array = np.array(data_array)

from sklearn.model_selection import train_test_split
train, test = train_test_split(data_array, test_size=0.25)
#train = data_array

users = int(max(user_info_dict.keys(), key=int)) + 1
movies = int(max(movie_info_dict.keys(), key=int)) + 1
total = 0
count = 0
user_ratings = [0 for i in range(users)]
movie_ratings = [0 for i in range(movies)]
user_ratings_count = [0 for i in range(users)]
movie_ratings_count = [0 for i in range(movies)]


user_preference_data = [[] for i in range(users)]
movie_preference_data = [[] for i in range(movies)]

for curr_row in train:
    curr_user = int(np.asscalar(curr_row[1]))
    curr_movie = int(np.asscalar(curr_row[2]))
    curr_rating = int(np.asscalar(curr_row[3]))

    user_ratings[curr_user] += curr_rating
    movie_ratings[curr_movie] += curr_rating
    user_ratings_count[curr_user] += 1
    movie_ratings_count[curr_movie] += 1

'''
    #Start with rating
    movie_item_to_append = []
    movie_item_to_append.append(curr_rating)
    #Append user info
    movie_item_to_append.extend(user_info_dict.get(curr_user))

    #Add to movie preference data for computation later
    movie_preference_data[curr_movie].append(movie_item_to_append)


    #Start with rating
    user_item_to_append = []
    user_item_to_append.append(curr_rating)
    #Append movie year
    user_item_to_append.extend(movie_info_dict.get(curr_row[2])[0:1])

    #Turn categorical values into binary values for movie categories
    #and append them
    split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
    for category in movie_categories_list:
        user_item_to_append.append(int(category in split_movie_items))

    #Add to user preference data for computation later
    user_preference_data[curr_user].append(user_item_to_append)

#movie_preference_data = np.array(movie_preference_data)
#user_preference_data = np.array(user_preference_data)
'''
avg_rating = np.mean(train[:, 3].astype(np.float))

user_deviation = calculate_deviations(user_ratings, user_ratings_count, avg_rating, users)
movie_deviation = calculate_deviations(movie_ratings, movie_ratings_count, avg_rating, movies)

'''
male_gender_diffs = [0 for i in range(movies)]
female_gender_diffs = [0 for i in range(movies)]

for i in range(movies):
    curr_movie_info = np.array(movie_preference_data[i])
    curr_movie_rating = avg_rating + movie_deviation[i]
    avg_male_diff = 0.0
    avg_female_diff = 0.0

    try:
        gender = curr_movie_info[:, 1]

        avg_male_rating = np.nanmean(curr_movie_info[gender == 'M', 0].astype(np.float))
        avg_female_rating = np.nanmean(curr_movie_info[gender == 'F', 0].astype(np.float))

        if not (np.isnan(avg_male_rating) or np.isnan(avg_female_rating)):
            avg_male_diff = avg_male_rating - avg_rating
            avg_female_diff = avg_female_rating - avg_rating

    except:
        pass

    #try:
    #    ages = curr_movie_info[:, 2]

    print(avg_male_diff)
    print(avg_female_diff)
    print("\n")

    male_gender_diffs[i] = avg_male_diff
    female_gender_diffs[i] = avg_female_diff

'''


#curr_user_movie_info











#Measure performance
test_ratings = test[:, 3]
test_data = test[:, :3]
ratings = predict(test_data, user_deviation, movie_deviation)
from sklearn import metrics
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(test_ratings,ratings)))

'''
#Debug output
print("average rating: {0}".format(avg_rating))
print(user_ratings[4557])
print(user_ratings_count[4557])
print(user_deviation[4557])
print(movie_ratings[3867])
print(movie_ratings_count[3867])
print(movie_deviation[3867])


transaction_IDs = []
test_array = []
with open('test.txt', 'r') as test_CSV:
    test_reader = csv.reader(test_CSV)

    #First row contains the format of the data
    test_format = next(test_reader)


    for curr_row in test_reader:
        test_array.append(curr_row)
        transaction_IDs.append(curr_row[0])
    #test_array = np.array(data_array)


rating_predictions = predict(test_array, user_deviation, movie_deviation)


with open('new_output.txt', 'w') as output_prediction_file:
        #Write header
        output_prediction_file.write(u"Id,rating")


        for i in range(len(transaction_IDs)):
            output_prediction_file.write(u"\n{},{}".format(transaction_IDs[i], rating_predictions[i]))
'''
