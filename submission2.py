#Compatibility with py2
from __future__ import division
from builtins import *

import csv
import numpy as np


from collections import Counter

def predict(test_array):
    ratings = []
    for item in test_array:
        #print(item)
        curr_user = int(np.asscalar(np.asarray(item[1])))
        curr_movie = int(np.asscalar(np.asarray(item[2])))
        curr_score = avg_rating + user_deviation[curr_user] * 0.75 + movie_deviation[curr_movie] * 0.75
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
    movie_info_dict = {}
    for curr_row in movie_info_reader:
        movie_info_dict[curr_row[0]] = curr_row[1:]

#Parse user info file
with open('user.txt', 'r') as user_CSV:
    user_info_reader = csv.reader(user_CSV)

    #First row contains header, no need to store it
    next(user_info_reader)

    #Get the values of the user information into a dictionary with the key being the user ID
    user_info_dict = {}
    for curr_row in user_info_reader:
        user_info_dict[curr_row[0]] = curr_row[1:]


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
train = data_array

users = int(max(user_info_dict.keys(), key=int)) + 1
movies = int(max(movie_info_dict.keys(), key=int)) + 1
total = 0
count = 0
user_ratings = [0 for i in range(users)]
movie_ratings = [0 for i in range(movies)]
user_ratings_count = [0 for i in range(users)]
movie_ratings_count = [0 for i in range(movies)]

for curr_row in train:
    curr_user = int(np.asscalar(curr_row[1]))
    curr_movie = int(np.asscalar(curr_row[2]))
    curr_rating = int(np.asscalar(curr_row[3]))
    user_ratings[curr_user] += curr_rating
    movie_ratings[curr_movie] += curr_rating
    user_ratings_count[curr_user] += 1
    movie_ratings_count[curr_movie] += 1
    total += curr_rating
    count += 1

avg_rating = total / count

user_deviation = [0 for i in range(users)]
movie_deviation = [0 for i in range(users)]
for i in range(users):
    try:
        user_avg = user_ratings[i] / user_ratings_count[i]
        user_deviation[i] = user_avg - avg_rating
    except:
        user_deviation[i] = 0.0

for i in range(movies):
    try:
        movie_avg = movie_ratings[i] / movie_ratings_count[i]
        movie_deviation[i] = movie_avg - avg_rating
    except:
        movie_deviation[i] = 0.0



test_ratings = test[:, 3]
test_data = test[:, :3]
ratings = predict(test_data)
from sklearn import metrics
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(test_ratings,ratings)))

print("average rating: {0}".format(avg_rating))
print(user_ratings[4557])
print(user_ratings_count[4557])
print(user_deviation[4557])
print(movie_ratings[3867])
print(movie_ratings_count[3867])
print(movie_deviation[3867])


#TODO implement output
transaction_IDs = []
test_array = []
with open('test.txt', 'r') as test_CSV:
    test_reader = csv.reader(test_CSV)

    #First row contains the format of the data
    test_format = next(test_reader)
    print(test_format)

    for curr_row in test_reader:
        test_array.append(curr_row)
        transaction_IDs.append(curr_row[0])
    #test_array = np.array(data_array)

#print(test_array[0])
rating_predictions = predict(test_array)
print(test_array[1])
print(rating_predictions[1])
with open('new_output.txt', 'w') as output_prediction_file:
        #Write header
        output_prediction_file.write(u"Id,rating")


        for i in range(len(transaction_IDs)):
            output_prediction_file.write(u"\n{},{}".format(transaction_IDs[i], rating_predictions[i]))
