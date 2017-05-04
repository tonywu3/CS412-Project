#Compatibility with python2
from __future__ import division
from builtins import *

import csv
import numpy as np

#Main code
def main():
    #Read movie info, user info, and training info files
    movie_info_dict, unique_years_set, movie_categories_list = read_movie_info("movie.txt")
    user_info_dict = read_user_info("user.txt")
    training_data_array = read_training_info("train.txt")

    #Find the maximum ID of users and movies to use in array indexing
    users_count = int(max(user_info_dict.keys(), key=int)) + 1
    movies_count = int(max(movie_info_dict.keys(), key=int)) + 1

    #Separate years into bins
    #requested_bin_count = 25
    #year_bins = get_equal_width_item_bins(unique_years_set, requested_bin_count)

    #Calculate average rating of all movies by users
    avg_rating = np.mean(training_data_array[:, 3].astype(np.float))

    #Calculate user deviations and movie deviations for each movie and user from the average rating
    user_ratings, user_ratings_count, movie_ratings, movie_ratings_count = \
            get_user_and_movie_ratings_counts(training_data_array, users_count, movies_count)

    user_deviations = calculate_deviations(user_ratings, user_ratings_count, avg_rating, users_count)
    movie_deviations = calculate_deviations(movie_ratings, movie_ratings_count, avg_rating, movies_count)

    #Get movie and user preference data
    #movie_preference_data, user_preference_data = get_user_and_movie_preferences(users_count, movies_count, training_data_array)

    #Calculate gender-based deviations per movie
    #male_gender_diffs, female_gender_diffs = get_per_movie_gender_rating_deviations(movies_count, movie_preference_data)

    #Read test file and generate predicted ratings
    test_array, transaction_IDs = read_test_info("test.txt")
    rating_predictions = predict(avg_rating, test_array, user_deviations, movie_deviations) #male_gender_diffs, female_gender_diffs)

    #Write predicted ratings to a file for Kaggle
    write_predictions(transaction_IDs, rating_predictions, "output.csv")



#Parse the movie info file
#Returns a dictionary of (movieID, movieInfo), a set of years, and a list of movie categories
def read_movie_info(movie_filename):
    unique_years_set = set()
    movie_info_dict = {}
    movie_categories = set()
    with open(movie_filename, 'r') as movie_CSV:
        movie_info_reader = csv.reader(movie_CSV)

        #First row contains header, no need to store it
        next(movie_info_reader)

        #Get the values of the movie information into a dictionary with the key being the movie ID
        #Also find unique categories

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

    return movie_info_dict, unique_years_set, movie_categories_list


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


#Parse the test info file
#Returns a list of transaction IDs and an array of (array of line elements split by a comma)
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


#Calculate rating sums and rating counts for each user and movie
#Returns four arrays with this data, the index of each array corresponding to
#the user or movie that they were calculated from
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

        #Add current rating to rating totals and increment rating counts for each user and movie
        user_ratings[curr_user] += curr_rating
        movie_ratings[curr_movie] += curr_rating
        user_ratings_count[curr_user] += 1
        movie_ratings_count[curr_movie] += 1

    return user_ratings, user_ratings_count, movie_ratings, movie_ratings_count


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


#Get more detailed information about each user and movie
#Returns two arrays of arrays of this data, indexed by movieID and userID for their corresponding arrays
def get_user_and_movie_preferences(users_count, movies_count, training_data_array):
    #Initialize empty arrays
    user_preference_data = [[] for i in range(users_count)]
    movie_preference_data = [[] for i in range(movies_count)]

        for curr_row in training_data_array:
            curr_user = int(np.asscalar(curr_row[1]))
            curr_movie = int(np.asscalar(curr_row[2]))
            curr_rating = int(np.asscalar(curr_row[3]))

            #Store info about which user characteristics (gender, age, occupation) rated which movie:
            #Start with rating
            movie_item_to_append = []
            movie_item_to_append.append(curr_rating)
            #Append user info
            movie_item_to_append.extend(user_info_dict.get(curr_user))

            #Add to movie preference data
            movie_preference_data[curr_movie].append(movie_item_to_append)

            #Store info about the characteristics of the movies (release year, genres) each user rated:
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

            #Add to user preference data
            user_preference_data[curr_user].append(user_item_to_append)

    return movie_preference_data, user_preference_data


#Calculate gender-based movie rating deviations
#Returns an array of male deviations and female deviations
def get_per_movie_gender_rating_deviations(movies_count, movie_preference_data):
    #Generate empty arrays
    male_gender_diffs = [0 for i in range(movies_count)]
    female_gender_diffs = [0 for i in range(movies_count)]

    #For each movie, calculate its gender-based deviations
    for i in range(movie_count):
        curr_movie_info = np.array(movie_preference_data[i])
        curr_movie_rating = avg_rating + movie_deviation[i]
        avg_male_diff = 0.0
        avg_female_diff = 0.0

        try:
            gender = curr_movie_info[:, 1]

            #Calculate the average female and male rating
            avg_male_rating = np.nanmean(curr_movie_info[gender == 'M', 0].astype(np.float))
            avg_female_rating = np.nanmean(curr_movie_info[gender == 'F', 0].astype(np.float))

            #Calculate the deviation in movie ranking based on gender,
            #making sure the average is not NaN
            if not (np.isnan(avg_male_rating) or np.isnan(avg_female_rating)):
                avg_male_diff = avg_male_rating - avg_rating
                avg_female_diff = avg_female_rating - avg_rating

        except:
            pass

        male_gender_diffs[i] = avg_male_diff
        female_gender_diffs[i] = avg_female_diff

    return male_gender_diffs, female_gender_diffs

#Write the predictions out to an output file in csv format
def write_predictions(transaction_IDs, rating_predictions, output_filename):
    with open(output_filename, 'w') as output_prediction_file:
            #Write header
            output_prediction_file.write(u"Id,rating")

            #Write predictions
            for i in range(len(transaction_IDs)):
                output_prediction_file.write(u"\n{},{}".format(transaction_IDs[i], rating_predictions[i]))




#Helper function to return the gender based on the userID
def get_gender(userID):
    return user_info_dict.get(userID)[0]


#Bin items based on equal-width binning
#Returns a list of lists of items, binned
def get_equal_width_item_bins(unique_items_set, requested_bin_count):
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
        if curr_bin_count >= unique_items_bin_count:
            curr_bin += 1
            curr_bin_count = 0

    #Prune array from empty bins:
    pruned_item_bins = [a for a in item_lists if a != []]

    return pruned_item_bins


#Bin items based on equal depth binning
#Returns a list of sets of items, binned
def get_equal_depth_item_bins(items_list, requested_bin_count):
    #Sort items and calculate the number of items that should be in each bin
    items_array_sorted = np.sort(np.array(items_list))
    items_count = len(items_list)
    items_bin_size = items_count / requested_bin_count

    items_sets = []
    curr_items_set = set()
    prev_items_set = set()
    items_bin_count = 0
    for item in items_array_sorted:

        #Make sure each set has no items in common with another set
        if item not in prev_items_set:
            curr_items_set.add(item)
            items_bin_count += 1
            if items_bin_count >= items_bin_size - 1:
                items_sets.append(curr_items_set)
                prev_items_set = curr_items_set
                items_bin_count = 0
                curr_items_set = set()

    return items_sets


#Generate predictions based on our classification technique
#Returns an array of integers of predicted ratings
def predict(avg_rating, test_array, user_deviation, movie_deviation): #male_gender_diffs, female_gender_diffs):
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

        #Calculate the expected rating
        curr_score = avg_rating + user_deviation[curr_user] * 0.75 + movie_deviation[curr_movie] * 0.75# + gender_offset

        #Round the expected rating to the nearest whole number
        ratings.append(int(round(curr_score)))

    ratings = np.array(ratings)

    return ratings


#Run main program
if __name__ == "__main__":
    main()


#Split data
'''
from sklearn.model_selection import train_test_split
train, test = train_test_split(data_array, test_size=0.25)
#train = data_array
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
