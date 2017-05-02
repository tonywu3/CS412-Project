import csv
import numpy as np

from collections import Counter
from itertools import product

from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


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

#movie_id = []
#Parse movie info file
with open('movie.txt', 'r') as movie_CSV:
    movie_info_reader = csv.reader(movie_CSV)

    #First row contains the format of the data
    movie_info_format = next(movie_info_reader)

    #Get the values of the movie information into a dictionary with the key being the movie ID
    #Also find unique categories
    movie_categories = set()
    movie_info_dict = {}
    for curr_row in movie_info_reader:
        #movie_id.append(curr_row[0])
        movie_categories.update(curr_row[2].split("|"))
        movie_info_dict[curr_row[0]] = curr_row[1:]

#Make sure order stays the same when used later
movie_categories_list = list(movie_categories)
#del movie_categories
movie_categories_list.remove('N/A')


#Parse user info file
with open('user.txt', 'r') as user_CSV:
    user_info_reader = csv.reader(user_CSV)

    #First row contains the format of the data
    user_info_format = next(user_info_reader)

    #Get the values of the user information into a dictionary with the key being the user ID
    user_info_dict = {}
    user_count = 0
    total_age = 0.0
    for curr_row in user_info_reader:
        user_count += 1
        user_info_dict[curr_row[0]] = curr_row[1:]

genre_list = []
#Parse training data file
with open('train.txt', 'r') as train_CSV:
    training_reader = csv.reader(train_CSV)

    #First row contains the format of the data
    training_format = next(training_reader)

    #Generate array of (array containing all the user and movie info)
    rating_info = []
    target_values = []
    for curr_row in training_reader:
        item_to_append = user_info_dict.get(curr_row[1]) + movie_info_dict.get(curr_row[2])[0:1]

        #TODO optimize or use another library to do this better, adds 4 seconds of processing time
        #Turn categorical values into numerical values for movie categories
        split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
        genre_list.append(split_movie_items)
        for category in movie_categories_list:
            item_to_append.append(int(category in split_movie_items))

        rating_info.append(item_to_append)
        target_values.append(curr_row[3])

rating_info = clean_data(np.array(rating_info))

target_values = np.array(target_values)

p_class = {}
value_counter = Counter(target_values)
for value in set(target_values):
    p_class[value] = ( value_counter[value] / float(len(target_values)) )

def conditional_probabilities(attribute_list, dictionary):
    temp_list = [ target_values, attribute_list ]
    temp_arr = np.array(temp_list)
    temp_counter = Counter(map(tuple, temp_arr.T))
    
    # P(attribute | class) = count(attribute | class) / count(class)
    for key in temp_counter.keys():
        temp_counter[key] /= float(value_counter[key[0]])
    dictionary.update(temp_counter)
        

p_gender = {}
conditional_probabilities(rating_info[:,0], p_gender)

p_age = {}
conditional_probabilities( rating_info[:,1], p_age)

p_occupation = {}
conditional_probabilities( rating_info[:,2], p_occupation)

p_year = {}
conditional_probabilities( rating_info[:,3], p_year)

#TODO calculate P(genre | class)
'''
genre_enc = LabelEncoder()
genre_enc.fit(movie_categories_list)
genres = genre_enc.transform(movie_categories_list)
'''
p_genre = {}




#conditional_probabilities( genres, p_genre)




'''

#Initial classification code:

#TODO Remove performance testing code before submitting final code
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(rating_info, target_values, test_size=0.15)

clf = tree.DecisionTreeClassifier(criterion='gini', max_depth=25, min_samples_leaf = 500)#criterion='entropy', max_depth=100, min_samples_leaf=1000)

#TODO Replace with full dataset
clf = clf.fit(rating_info, target_values)
#clf = clf.fit(X_train, y_train)

#TODO remove below code as well
from sklearn import metrics
def measure_performance(X,y,clf):
    y_pred=clf.predict(X)
    print ("Accuracy:{0:.3f}".format(metrics.accuracy_score(y,y_pred)))
    tree.export_graphviz(clf, out_file="tree.dot")

#measure_performance(X_test, y_test, clf)

#Prediction:
#Parse test data file
with open('test.txt', 'r') as test_CSV:
    test_reader = csv.reader(test_CSV)

    #First row contains the format of the data
    test_format = next(test_reader)

    #Generate array of (array containing all the user and movie info)
    transaction_IDs = []
    test_info = []
    for curr_row in test_reader:
        item_to_append = user_info_dict.get(curr_row[1]) + movie_info_dict.get(curr_row[2])[0:1]

        #TODO optimize or use another library to do this better
        #Turn categorical values into numerical values for movie categories
        split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
        for category in movie_categories_list:
            item_to_append.append(int(category in split_movie_items))

        test_info.append(item_to_append)
        transaction_IDs.append(curr_row[0])

test_info = clean_data(np.array(test_info))

rating_predictions = clf.predict(test_info)

#Write output file
with open('output.txt', 'w') as output_prediction_file:
    output_prediction_file.write("Id,rating")
    for i in range(len(rating_predictions)):
        output_prediction_file.write("\n{},{}".format(transaction_IDs[i], rating_predictions[i]))
'''

