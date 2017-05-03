import csv
import numpy as np

from collections import Counter
from itertools import product

from sklearn import tree
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

avg_age = 0
avg_yr = 0
def clean_data(list_to_clean, gender_column=1, age_column=2, occupation_column=3, year_column=4):
    global avg_age
    global avg_yr
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
    avg_age = str(average_age)
    #Replace missing occupations with -1
    list_to_clean[list_to_clean[:, occupation_column] == 'N/A', occupation_column] = -1

    #Find average year
    years = list_to_clean[:, year_column]
    average_year = int(np.mean(list_to_clean[years != 'N/A', year_column].astype(np.int)))
    #Replace missing years with average year
    list_to_clean[list_to_clean[:, year_column] == 'N/A', year_column] = average_year
    avg_yr = str(average_year)

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

movieid_list = []
#Parse training data file
with open('train.txt', 'r') as train_CSV:
    training_reader = csv.reader(train_CSV)

    #First row contains the format of the data
    training_format = next(training_reader)

    #Generate array of (array containing all the user and movie info)
    rating_info = []
    target_values = []
    for curr_row in training_reader:
        item_to_append = list(map(int, [curr_row[1]])) + user_info_dict.get(curr_row[1]) + movie_info_dict.get(curr_row[2])[0:1]
        movieid_list.append(curr_row[2])
        #TODO optimize or use another library to do this better, adds 4 seconds of processing time
        #Turn categorical values into numerical values for movie categories
        split_movie_items = movie_info_dict.get(curr_row[2])[1].split("|")
        #genre_list.append(split_movie_items)
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

common_gender = Counter(rating_info[:,1]).most_common(1)[0][0]
for value in user_info_dict.values():
    if value[0] == 'M':
        value[0] = '1'
    if value[0] == "F":
        value[0] = '0'
    if value[0] == "N/A":
        value[0] = common_gender
    if value[1] == "N/A":
        value[1] = avg_age
    if value[2] == "N/A":
        value[2] = '-1'

for value in movie_info_dict.values():
    if value[0] == 'N/A':
        value[0] = avg_yr
        
userid_set = list(set(rating_info[:,0]))
def conditional_probabilities(attribute_list, dictionary, colmn, userormovie):
    temp_list = [ rating_info[:,0], attribute_list, target_values ]
    temp_arr = np.array(temp_list)
    temp_counter = Counter(map(tuple,temp_arr.T))
    
    ret_dict = {}
    
    att_set = list(set(rating_info[:,colmn]))
    #ats.append(att_set)
    
    for i in userid_set:
        ret_dict[i] = {}
    
    conditions = [x for x in product(userid_set, att_set, sorted(p_class.keys()))]
    #cdl.append(conditions)
    
    for c in conditions:
        #Avoid 0 probability by +1 to all counts
        if userormovie == 'user':
            if c[1] == user_info_dict[c[0]][colmn-1]:
                ret_dict[c[0]].update( {(c[1],c[2]): (temp_counter[c]+1)/ float(value_counter[c[2]]) } )
        if userormovie == 'movie':
            try:
                if c[1] == movie_info_dict[c[0]][0]:
                    ret_dict[c[0]].update( {(c[1],c[2]): (temp_counter[c]+1)/ float(value_counter[c[2]]) } )
            except KeyError:
                ret_dict[c[0]].update( {('-1', '-1'): 1.} )
                
    dictionary.update(ret_dict)
        
    '''
    if(colmn != 4):
        temp_list = [ rating_info[:,0], target_values, attribute_list ]
    else:
        temp_list = [ movieid_list, target_values, attribute_list]
        
    temp_arr = np.array(temp_list)
    temp_counter = Counter(map(tuple, temp_arr.T))
    #if( colmn != 4):
    ret_dict = {}
    if( colmn != 4):
        for i in set(rating_info[:,0]):
            ret_dict[i] = {}
    else:
        for i in set(movieid_list):
            ret_dict[i] = {}
    # P(attribute | class) = count(attribute | class) / count(class)
    for key in temp_counter.keys():
        temp_counter[key] /= float(value_counter[key[1]])
        ret_dict[key[0]][ (key[1], key[2]) ] = temp_counter[key]
    #ret_dict format is ret_dict[userid] = rating, attribute
    dictionary.update(ret_dict)
    #else:
    #    for key in temp_counter.keys():
    #        temp_counter[key] /= float(value_counter[key[0]])
    #    dictionary.update(temp_counter)   
    '''

# Male = 1, Female = 0
p_gender = {}
conditional_probabilities(rating_info[:,1], p_gender, 1, 'user')

p_age = {}
conditional_probabilities( rating_info[:,2], p_age, 2, 'user')

p_occupation = {}
conditional_probabilities( rating_info[:,3], p_occupation, 3, 'user')

#TODO fix zero-probability for attribute
p_userid = {}
for u in user_info_dict.keys():
    #calculate gender classes for user
    p_userid[u] = {}
    for i in p_class.keys():
        p_userid[u][i] = {}
        running_prob = 1.        
        #Gender
            #Only select those probabilities that matches user gender
        running_prob *= p_gender[u][ sorted(p_gender[u])[int(i)-1] ]
        #Age
            #Only select those probs that matches user age
        running_prob *= p_age[u][ sorted(p_age[u])[int(i)-1] ]
        #Occupation
        running_prob *= p_occupation[u][ sorted(p_occupation[u])[int(i)-1] ]

        p_userid[u][i] = running_prob        
        


p_year = {}
conditional_probabilities( rating_info[:,4], p_year, 4, 'movie')

p_genres = {}
column = 5

for category in movie_categories_list:
    p_genres[category] = {}
    p_category = {}
    cat_list = [ rating_info[:,column], target_values ]
    cat_list = np.array(cat_list)
    cat_counter = Counter(map(tuple, cat_list.T))
    
    check_list = []
    for i in sorted(p_class.keys()):
        check_list.append ( ('1', i) )
        p_category[i] = cat_counter[('1', i)] / float(value_counter[str(i)])
            
    p_genres[category].update(p_category)
    column +=1
    

def predict(userid, movieid ):
    predict_list = []
    for i in sorted(p_class.keys()):
        prob = 1.
        prob *= p_userid[userid][i]
        
        if movieid in movie_info_dict.keys():
            year_keys = sorted(p_year[ movie_info_dict[movieid][0] ].keys())
            if( len(year_keys) > 1):
                prob *= p_year[ movie_info_dict[movieid][0] ][year_keys[int(i)-1]]
            #except IndexError:
            #    print movieid
            
            split_catlist = movie_info_dict[movieid][1].split("|")
            for cat in split_catlist:
                if(cat == "N/A"):
                    continue
                prob *= p_genres[cat][i]
        predict_list.append(prob)
        
    return predict_list.index(max(predict_list)) + 1


output = []
with open ("test.txt", "r") as test_CSV:
    test_reader = csv.reader(test_CSV)
    test_format = next(test_reader)
    
    for curr_row in test_reader:
        output.append( [curr_row[0], predict(curr_row[1], curr_row[2])] )


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

