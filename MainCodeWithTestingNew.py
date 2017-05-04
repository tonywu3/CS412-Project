#Compatibility with python2
from __future__ import division
from builtins import *

import csv
import numpy as np


#Helper function to return the gender based on the userID
def get_gender(userID):
    return user_info_dict.get(userID)[0]


#Calculates the deviation from the mean rating for each item in the array
#Returns an array of deviations
def calculate_deviations(total_array, counts_array, avg, size):
    #Generate empty array of the right size
    deviation = [0 for i in range(size)]

    #For each item, calculate its deviation if possible and store it in the corresponding place in the array
    for i in range(size):
        try:
            curr_avg = total_array[i] / counts_array[i]
            deviation[i] = curr_avg - avg
        except:
            deviation[i] = 0.0

    return deviation

#Bin items based on equal-width binning
#Returns a list of lists of items, binned
def get_item_bins(unique_items_set, requested_bin_count):
    #Calculate the number of items that should be in each bin
    unique_items_count = len(unique_items_set)
    unique_items_bin_count = unique_items_count / requested_bin_count

    #Put items in bins
    curr_bin = 0
    curr_bin_count = 0
    #Create empty array of arrays
    item_lists = [[] for i in range(requested_bin_count + 1)]
    #Put items in bins in sorted order
    for item in sorted(unique_items_set):
        item_lists[curr_bin].append(item)
        curr_bin_count += 1
        if (curr_bin_count >= unique_items_bin_count):
            curr_bin += 1
            curr_bin_count = 0

    #Prune array from empty bins:
    pruned_item_bins = [a for a in item_lists if a != []]

    return pruned_item_bins

#Generate predictions based on our classification technique
#Returns an array of integers of predicted ratings
def predict(test_array, user_deviation, movie_deviation, avg_rating):
    ratings = []
    for item in test_array:
        #Convert to int in a way that works with python2 and 3
        curr_user = int(np.asscalar(np.asarray(item[1])))
        curr_movie = int(np.asscalar(np.asarray(item[2])))

        #Experimental gender weighting
        '''
        curr_gender = get_gender(curr_user)
        gender_offset = 0.0
        if (curr_gender == 'M'):
            gender_offset = male_gender_diffs[curr_movie]
        elif (curr_gender == 'F'):
            gender_offset = female_gender_diffs[curr_movie]
        '''

        curr_score = avg_rating + user_deviation[curr_user] * 0.75 + movie_deviation[curr_movie] * 0.75 #+ gender_offset

        ratings.append(int(round(curr_score)))

    ratings = np.array(ratings)

    return ratings


#Parse the movie info file
#Returns a dictionary of (movieID, movieInfo), a set of years, and a list of movie categories
def read_movie_info(movie_filename):
    unique_years_set = set()
    movie_info_dict = {}
    with open(movie_filename, 'r') as movie_CSV:
        movie_info_reader = csv.reader(movie_CSV)

        #First row contains header, no need to store it
        next(movie_info_reader)

        #Get the values of the movie information into a dictionary with the key being the movie ID
        #Also find unique categories
        movie_categories = set()
        for curr_row in movie_info_reader:
            movie_categories.update(curr_row[2].split("|"))
            movie_info_dict[curr_row[0]] = curr_row[1:]
            curr_year = curr_row[1]
            if not (curr_year == '1' or curr_year == "N/A"):
                unique_years_set.add(curr_year)

        #Make sure order of categories stays the same when used later by making it a list
        movie_categories_list = list(movie_categories)
        movie_categories_list.remove("N/A")

        #Prevent accidental usage of unordered set later on
        del movie_categories

    return movie_info_dict, unique_years_set


#Parse user info file
#Return dictionary of (userID, user info)
def read_user_info(user_filename):
    user_info_dict = {}
    with open(user_filename, 'r') as user_CSV:
        user_info_reader = csv.reader(user_CSV)

        #First row contains header, no need to store it
        next(user_info_reader)

        #Get the values of the user information into a dictionary with the key being the user ID
        for curr_row in user_info_reader:
            user_info_dict[int(curr_row[0])] = curr_row[1:]

    return user_info_dict


#Parse training data file
#Return array of (array of line elements split by a comma)
def read_training_info(training_filename):
    data_array = []
    with open(training_filename, 'r') as train_CSV:
        training_reader = csv.reader(train_CSV)

        #First row contains header, no need to store it
        next(training_reader)

        #Read in data
        for curr_row in training_reader:
            data_array.append(curr_row)
        data_array = np.array(data_array)

    return data_array



'''
movie_years_array_sorted = np.sort(np.array(movie_years_array))
movie_years_count = len(movie_years_array)
movie_years_bin_size = movie_years_count / 20.0
print(movie_years_bin_size)

movie_years_sets = []
curr_movie_years_set = set()
prev_movie_years_set = set()
movie_years_bin_count = 0
for year in movie_years_array_sorted:
    if year not in prev_movie_years_set:
        curr_movie_years_set.add(year)
        movie_years_bin_count += 1
        if movie_years_bin_count >= movie_years_bin_size - 1:
            movie_years_sets.append(curr_movie_years_set)
            prev_movie_years_set = curr_movie_years_set
            movie_years_bin_count = 0
            curr_movie_years_set = set()

print(movie_years_sets)

'''
'''
        #Save the valid ages in another array
        curr_age = curr_row[2]
        if not (curr_age == '1' or curr_age == "N/A"):
            ages_array.append(curr_row[2])

ages_array_sorted = np.sort(np.array(ages_array))
ages_count = len(ages_array)
ages_bucket_size = ages_count / 3.0
print(ages_bucket_size)

ages_sets = []
curr_ages_set = set()
ages_bucket_count = 0
for age in ages_array_sorted:
    curr_ages_set.add(age)
    ages_bucket_count += 1
    if ages_bucket_count >= ages_bucket_size - 1:
        ages_sets.append(curr_ages_set)
        ages_bucket_count = 0
        curr_ages_set = set()

print(ages_sets)
'''




def read_test_info(test_filename):
    transaction_IDs = []
    test_array = []
    with open(test_filename, 'r') as test_CSV:
        test_reader = csv.reader(test_CSV)

        #First row contains the format of the data, no need to store it
        next(test_reader)

        #Read and store test data and transaction IDs, keeping order the same
        for curr_row in test_reader:
            test_array.append(curr_row)
            transaction_IDs.append(curr_row[0])

    return test_array, transaction_IDs


def get_user_and_movie_ratings_counts(training_data_array, users_count, movies_count):
    #Initialize empty arrays
    user_ratings = [0 for i in range(users_count)]
    movie_ratings = [0 for i in range(movies_count)]
    user_ratings_count = [0 for i in range(users_count)]
    movie_ratings_count = [0 for i in range(movies_count)]


    for curr_row in training_data_array:
        curr_user = int(np.asscalar(curr_row[1]))
        curr_movie = int(np.asscalar(curr_row[2]))
        curr_rating = int(np.asscalar(curr_row[3]))

        user_ratings[curr_user] += curr_rating
        movie_ratings[curr_movie] += curr_rating
        user_ratings_count[curr_user] += 1
        movie_ratings_count[curr_movie] += 1

    return user_ratings, user_ratings_count, movie_ratings, movie_ratings_count
'''
    user_preference_data = [[] for i in range(users)]
    movie_preference_data = [[] for i in range(movies)]
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

    print(avg_male_diff)
    print(avg_female_diff)
    print("\n")

    male_gender_diffs[i] = avg_male_diff
    female_gender_diffs[i] = avg_female_diff

return male_gender_diffs, female_gender_diffs

'''

#Measure performance
'''
test_ratings = test[:, 3]
test_data = test[:, :3]
ratings = predict(test_data, user_deviation, movie_deviation)
ratings = np.array(ratings).astype('str')
from sklearn import metrics
print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(test_ratings,ratings)))
'''

#Debug output
'''
print("average rating: {0}".format(avg_rating))
print(user_ratings[4557])
print(user_ratings_count[4557])
print(user_deviation[4557])
print(movie_ratings[3867])
print(movie_ratings_count[3867])
print(movie_deviation[3867])
'''
'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(data_array, test_size=0.25)
#train = data_array
'''


def write_predictions(transaction_IDs, rating_predictions, output_filename):
    with open(output_filename, 'w') as output_prediction_file:
            #Write header
            output_prediction_file.write(u"Id,rating")

            #Write predictions
            for i in range(len(transaction_IDs)):
                output_prediction_file.write(u"\n{},{}".format(transaction_IDs[i], rating_predictions[i]))



#Main code
def main():
    #Read movie info, user info, and training info files
    movie_info_dict, unique_years_set = read_movie_info("movie.txt")
    user_info_dict = read_user_info("user.txt")
    training_data_array = read_training_info("train.txt")

    #Find the maximum ID of users and movies to use in array indexing
    users_count = int(max(user_info_dict.keys(), key=int)) + 1
    movies_count = int(max(movie_info_dict.keys(), key=int)) + 1

    #Separate years into bins
    requested_bin_count = 25
    year_bins = get_item_bins(unique_years_set, requested_bin_count)

    #Calculate average rating of all movies by users
    avg_rating = np.mean(training_data_array[:, 3].astype(np.float))

    #Calculate user deviations and movie deviations for each movie and user from the average rating
    user_ratings, user_ratings_count, movie_ratings, movie_ratings_count = \
            get_user_and_movie_ratings_counts(training_data_array, users_count, movies_count)

    user_deviations = calculate_deviations(user_ratings, user_ratings_count, avg_rating, users_count)
    movie_deviations = calculate_deviations(movie_ratings, movie_ratings_count, avg_rating, movies_count)

    #Read test file and generate predicted ratings
    test_array, transaction_IDs = read_test_info("test.txt")
    rating_predictions = predict(test_array, user_deviations, movie_deviations, avg_rating)

    #Write predicted ratings to a file for Kaggle
    write_predictions(transaction_IDs, rating_predictions, "output.csv")



#Run main program
if __name__ == "__main__":
    main()
