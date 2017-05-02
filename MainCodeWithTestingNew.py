import csv
import numpy as np

from collections import Counter



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


users = int(max(user_info_dict.keys(), key=int)) + 1
movies = int(max(movie_info_dict.keys(), key=int)) + 1
total = 0
count = 0
user_ratings = [0 for i in range(users)]
movie_ratings = [0 for i in range(movies)]
user_ratings_count = [0 for i in range(users)]
movie_ratings_count = [0 for i in range(movies)]

for curr_row in train:
    curr_user = int(curr_row[1])
    curr_movie = int(curr_row[2])
    curr_rating = int(curr_row[3])
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

ratings = []
for item in test_data:
    curr_score = avg_rating + user_deviation[int(item[1])] + movie_deviation[int(item[2])]
    ratings.append(str(int(round(curr_score, 0))))

ratings = np.array(ratings)


from sklearn import metrics
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(test_ratings,ratings)))


#TODO implement output
