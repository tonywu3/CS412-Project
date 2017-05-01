import csv
import numpy as np

from collections import Counter

from sklearn import tree
from sklearn.preprocessing import LabelEncoder


def clean_data(list_to_clean, gender_column=0, age_column=1, occupation_column=2, year_column=3):
    #Find majority gender
    genders = list_to_clean[:, gender_column]
    data = Counter(genders)
    most_common_gender = data.most_common(1)[0][0]
    if (most_common_gender == "N/A"):
        most_common_gender = data.most_common(2)[0][0]

    #Replace missing genders with majority gender
    list_to_clean[list_to_clean[:, gender_column] == 'N/A', gender_column] = most_common_gender

    #Replace gender with integer value
    enc = LabelEncoder()
    label_encoder = enc.fit(list_to_clean[:, gender_column])
    transformed_genderinfo = label_encoder.transform(list_to_clean[:, gender_column])
    list_to_clean[:, gender_column] = transformed_genderinfo


    #Find average age
    ages = list_to_clean[:, age_column]
    average_age = int(np.mean(list_to_clean[ages != 'N/A', age_column].astype(np.int)))

    #Replace missing ages with average age
    list_to_clean[list_to_clean[:, age_column] == 'N/A', age_column] = average_age


    #Replace missing occupations with -1
    list_to_clean[list_to_clean[:, occupation_column] == 'N/A', occupation_column] = -1


    #Find average year
    years = list_to_clean[:, year_column]
    average_year = int(np.mean(list_to_clean[years != 'N/A', year_column].astype(np.int)))

    #Replace missing years with average year
    list_to_clean[list_to_clean[:, year_column] == 'N/A', year_column] = average_year


    return list_to_clean


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
        user_info_dict[curr_row[0]] = curr_row[1:]


#Parse training data file
with open('train.txt', 'r') as train_CSV:
    training_reader = csv.reader(train_CSV)

    #First row contains header, no need to store it
    next(training_reader)

    #Generate array of (array containing all the user and movie info)
    #Format: [gender, age, occupation, year, (category existence entries)]
    rating_info = []
    target_values = []
    for curr_row in training_reader:
        item_to_append = user_info_dict.get(curr_row[1]) + movie_info_dict.get(curr_row[2])[0:1]

        #Turn categorical values into binary values for movie categories
        split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
        for category in movie_categories_list:
            item_to_append.append(int(category in split_movie_items))

        rating_info.append(item_to_append)
        target_values.append(curr_row[3])

#Clean data and transform it into numpy arrays
rating_info = np.array(rating_info)
rating_info = clean_data(rating_info)
target_values = np.array(target_values)

#Classification code:
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rating_info, target_values, test_size=0.15)

#Generate decision tree with a max depth of 25 and a
#minimum number of samples per leaf node of 500
clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=25, min_samples_leaf = 500)

#Train decision tree
#TODO Replace with full dataset
#clf = clf.fit(rating_info, target_values)
clf = clf.fit(X_train, y_train)

from sklearn import metrics
def measure_performance(X,y,clf):
    y_pred=clf.predict(X)
    print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)))
    tree.export_graphviz(clf, out_file="tree.dot")

measure_performance(X_test, y_test, clf)

#Prediction:
#Parse test data file
with open('test.txt', 'r') as test_CSV:
    test_reader = csv.reader(test_CSV)

    #First row contains the format of the data
    test_format = next(test_reader)

    #Generate array of (array containing all the user and movie info)
    #Format: [gender, age, occupation, year, (category existence entries)]
    transaction_IDs = []
    test_info = []
    for curr_row in test_reader:
        item_to_append = user_info_dict.get(curr_row[1]) + movie_info_dict.get(curr_row[2])[0:1]

        #Turn categorical values into binary values for movie categories
        split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
        for category in movie_categories_list:
            item_to_append.append(int(category in split_movie_items))

        test_info.append(item_to_append)
        transaction_IDs.append(curr_row[0])

#Clean data and transform it into numpy arrays
test_info = np.array(test_info)
test_info = clean_data(test_info)

#Calculate predicted values
rating_predictions = clf.predict(test_info)

#Write output file
with open('output.txt', 'w') as output_prediction_file:
    #Write header
    output_prediction_file.write("Id,rating")

    #Write predictions
    for i in range(len(rating_predictions)):
        output_prediction_file.write("\n{},{}".format(transaction_IDs[i], rating_predictions[i]))








#########################################
