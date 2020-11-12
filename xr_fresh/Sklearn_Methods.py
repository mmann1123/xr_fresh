import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
import pandas as pd

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

def create_name_parameter_to_classifier_mapping(classif_dict=None, tuned_parameters_dict=None):
    """
    Creates some classifiers and some parameters in case none is provided
    :param classif_dict: None or dictionary of {name_classif:classifier}
    :param tuned_parameters_dict: None or dictionary of {name_classif:parameters_lists}
    :return: classif_dict, tuned_parameters_dict
    """
    if classif_dict is None:
        classif_dict ={
                    # "NearestNeighbors": KNeighborsClassifier(n_neighbors= 7),
                    #    "LinearSVM": LinearSVC(random_state=0, tol=1e-5, C=10),
                       # "DecisionTree": DecisionTreeClassifier(max_depth=5, min_samples_leaf= 4),
                       # "RandomForest": RandomForestClassifier(max_features=5,max_depth= 10, n_estimators= 200),
                       "NeuralNet": MLPClassifier(alpha = 0.001, hidden_layer_sizes = (16,)),
                        "XgBoost":XGBClassifier()
                       # "AdaBoost": AdaBoostClassifier(n_estimators= 100)
                       }

    ## parameters of the above classifiers to tune on
    if tuned_parameters_dict is None:
        tuned_parameters_dict = {
                                # "NearestNeighbors": {'n_neighbors': [3, 5, 7]},
                                #  "LinearSVM": {'C': [1, 10, 100, 1000]},
                                #  "DecisionTree": {'min_samples_leaf': [1, 2, 4]},
                                # "RandomForest": {'max_depth': [100],
                                #  'max_features': [5],
                                #  'min_samples_leaf': [10],
                                #  # 'min_samples_split': [8],
                                #  'n_estimators': [300]},
                                 "NeuralNet": {'hidden_layer_sizes' :[(64,), (32,32), (100,)],'max_iter':[250, 500]},
                                 "XgBoost"  : {
                                                'min_child_weight': [1, 5, 10],
                                                'gamma': [0.5, 1, 1.5, 2, 5],
                                                'subsample': [0.6, 0.8, 1.0],
                                                'colsample_bytree': [0.6, 0.8, 1.0],
                                                'max_depth': [3, 4, 5]
                                            }
                                 # "AdaBoost": {'n_estimators': [50, 100]}
                                 }

    return classif_dict, tuned_parameters_dict



def build_and_compare_classifiers_with_gridsearch(X_train, y_train,selected_features,folder_saving,
                                                  classif_dict=None, tuned_parameters_dict=None):
    """
    Comparing classifiers with a grid search over the model parameters.
    :return: dictionary with mapping of each classifier with it's best estimated parameters
    """
    f = open(folder_saving+'results.txt', 'w')

    classif_dict, tuned_parameters_dict = \
        create_name_parameter_to_classifier_mapping(classif_dict, tuned_parameters_dict)

    max_score = -1
    best_estimator_for_all_classifiers = {}
    best_clf = None

    X_train = X_train[selected_features]


    for name in classif_dict.keys():
        print(name)
        clf = classif_dict[name]
        ## automatic stratified kfold by grid search
        grid_search_clf = RandomizedSearchCV(clf, tuned_parameters_dict[name], scoring='roc_auc')
        grid_search_clf.fit(X_train, y_train)

        print('best mean score for ' + name + '=' + str(grid_search_clf.best_score_))
        f.write('best mean score for ' + name + '=' + str(grid_search_clf.best_score_)+'\n')
        f.write('best parameters for ' + name + '=' + str(grid_search_clf.best_params_) + '\n')
        f.write('\n')
        print(grid_search_clf.best_params_)
        best_estimator_for_all_classifiers[name] = grid_search_clf.best_estimator_

        ## finding the best classifier wrt to grid search's score (not using it anywhere)
        if grid_search_clf.best_score_ > max_score:
            max_score = grid_search_clf.best_score_
            best_clf = grid_search_clf.best_estimator_


    f.close()

    print(best_clf)

    return best_estimator_for_all_classifiers







