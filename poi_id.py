#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

import matplotlib.pyplot
import numpy as np
from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import SelectKBest
from sklearn.tree import DecisionTreeClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


### Load the dictionary containing the dataset
with open("final_project_dataset.pkl", "r") as data_file:
    data_dict = pickle.load(data_file)


### Remove outliers
data_dict.pop("TOTAL", 0)

### Create new feature(s)
### return number of poi messages to/from as proportion of all messages to/from
def computeProportion( poi_messages, all_messages ):
    if poi_messages == "NaN" or all_messages == "NaN":
        proportion = 0
    else:
        proportion = float(poi_messages) / float(all_messages)

    return proportion

### enter poi message proportions for each employee in data_dict
for name in data_dict:

    data_point = data_dict[name]

    from_poi_to_this_person = data_point["from_poi_to_this_person"]
    to_messages = data_point["to_messages"]
    proportion_from_poi = computeProportion( from_poi_to_this_person, to_messages )
    data_point["proportion_from_poi"] = proportion_from_poi

    from_this_person_to_poi = data_point["from_this_person_to_poi"]
    from_messages = data_point["from_messages"]
    proportion_to_poi = computeProportion( from_this_person_to_poi, from_messages )
    data_point["proportion_to_poi"] = proportion_to_poi

### Store to my_dataset for easy export below.
my_dataset = data_dict

features_list = ['poi', 'salary', 'deferral_payments', 'total_payments', \
                 'bonus', 'restricted_stock_deferred', 'deferred_income', \
                 'total_stock_value', 'expenses', 'exercised_stock_options', \
                 'other', 'long_term_incentive', 'restricted_stock', \
                 'director_fees', 'proportion_from_poi', 'proportion_to_poi']

### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list)
labels, features = targetFeatureSplit(data)
features_train, features_test, labels_train, labels_test = \
    train_test_split(features, labels, test_size = 0.3, random_state = 42)

### Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

### Task 5: Tune your classifier to achieve better than .3 precision and recall
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info:
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

pipe = Pipeline([
                ('scaling', MinMaxScaler()),
                ('skb', SelectKBest()),
                ('tree', DecisionTreeClassifier(random_state = 42))
                ])


parameters = {
    'skb__k': [3, 4, 5],
    'tree__criterion': ['gini', 'entropy'],
    'tree__max_depth': [2, 3, 4],
    'tree__class_weight': ['balanced', None]
}
strat_shuff = StratifiedShuffleSplit(labels, 100, random_state = 42, \
                                        test_size = 0.3)

gs = GridSearchCV(pipe, param_grid = parameters, scoring = 'f1', \
                    cv = strat_shuff)

gs.fit(features, labels)

### show best estimator parameters
print ""
print "Best estimator found by grid search: "
print gs.best_estimator_

clf = gs.best_estimator_
clf.fit(features_train, labels_train)

tree_pred = clf.predict(features_test)

### list important features for DecisionTree by rank
d_tree = clf.named_steps["tree"]
d_tree_imp = d_tree.feature_importances_
indices = np.argsort(d_tree_imp)[::-1]
print ""
print "DecisionTree Feature Ranking: "
for i in range(3):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1], \
                d_tree_imp[indices[i]])

### list best features for SelectKBest by rank
k_best = clf.named_steps["skb"]
k_best_scr = k_best.scores_
indices = np.argsort(k_best_scr)[::-1]
print ""
print "SelectKBest Feature Ranking: "
for i in range(5):
    print "feature no. {}: {} ({})".format(i+1,features_list[indices[i]+1], \
                k_best_scr[indices[i]])


accuracy = accuracy_score(tree_pred, labels_test)
precision = precision_score(tree_pred, labels_test)
recall = recall_score(tree_pred, labels_test)

print ""
print "Accuracy Score: ", accuracy
print "Precision Score: ", precision
print "Recall Score: ", recall


### Task 6: Dump your classifier, dataset, and features_list so anyone can
### check your results. You do not need to change anything below, but make sure
### that the version of poi_id.py that you submit can be run on its own and
### generates the necessary .pkl files for validating your results.

pickle.dump(clf, open("my_classifier.pkl", "w") )
pickle.dump(my_dataset, open("my_dataset.pkl", "w") )
pickle.dump(features_list, open("my_feature_list.pkl", "w") )
dump_classifier_and_data(clf, my_dataset, features_list)
