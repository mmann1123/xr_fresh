import os
import datetime
import pandas as pd
import numpy as np
import Sklearn_Methods
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score,precision_score,f1_score,recall_score

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)


path = "/Users/saumya/Desktop/Wildfire_sample_project/"

## sample data file
csv_file = "TestData_1970_2019.csv"
## selected features to be used for training (corresponds to the readme - I added "Elev" and saw better results)
selected_features = ['City_Bounds', 'Cultivated_Prop', 'Elec_Dist', 'Elev','Mean_Housing_Dens_25km', 'Road_Dist', 'aet__mean_Normal', 'cwd__mean_Normal', 'aet__mean_ThreeYear_Dev', 'cwd__mean_ThreeYear_Dev']
## Y (target)
target = "Fire"

essential_cols = ["pixel_id", "year", "Fire"]

def read_data(csv_file):

    ## reading the csv file into a pandas dataframe
    input_df = pd.read_csv(csv_file, index_col=0)
    print("% of positive labels(fire): ",input_df.Fire.sum()/len(input_df))
    print(input_df.describe())

    columns = list(input_df.columns)
    print("Total num of columns: ",len(columns))
    print("Check Nans: ",input_df.isnull().values.any())

    return input_df, columns


def get_train_test_data(df):
    '''
    Create the final train, test and valid datasets
    '''

    ## keeping only the selected features with "pixel_id", "year", "Fire"
    X = df[essential_cols+selected_features]
    y = df[target]

    ## dividing data into train(70%)/valid(15/5)/test(15%) -- stratified split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,random_state=42, stratify=y)
    X_valid, X_test, y_valid, y_test = train_test_split(X_test, y_test, test_size=0.5, random_state=42, stratify=y_test)

    print('train data shape: ',X_train.shape, y_train.shape)
    print('valid data shape: ', X_valid.shape, y_valid.shape)
    print('test data shape: ', X_test.shape, y_test.shape)

    # print("% of positive samples in train data: ", y_train.sum()/len(y_train))
    # print("% of positive samples in valid data: ", y_valid.sum()/len(y_valid))
    # print("% of positive samples in test data: ", y_test.sum()/len(y_test))

    return X_train, X_valid, X_test, y_train, y_valid, y_test



def standardize_from_train(X_train, X_valid=None, X_test=None):
    """
    Standardize (or 'normalize') the feature matrices based on X_train.
    """

    for col in selected_features:
        mean = X_train[col].mean()
        std = X_train[col].std()

        X_train[col] = (X_train[col] - mean) / std
        X_valid[col] = (X_valid[col] - mean) / std
        X_test[col] = (X_test[col] - mean) / std

    print(X_train.describe())

    return X_train, X_valid, X_test



def results_on_valid_test_dataset_with_gridsearch(best_estimator_for_all_classifiers, X_valid, y_valid, X_test, y_test, X_train,
                                  y_train, folder_saving):
    """
    Writing the classifier predictions for both the validation and test datasets.
    :param best_estimator_for_all_classifiers: best estimators obtained from grid search for each classifier
     """

    f = open(folder_saving + 'results.txt', 'a')

    selected_cols = essential_cols

    dataset_valid = X_valid
    dataset_test = X_test

    ## only using the selected features for training
    X_train = X_train[selected_features]
    X_valid = X_valid[selected_features]
    X_test = X_test[selected_features]

    for clf in best_estimator_for_all_classifiers:
        best_clf = best_estimator_for_all_classifiers[clf].fit(X_train, y_train)

        # check if prediction probabilities is supported by the particular clf
        if hasattr(best_clf, 'predict_proba'):
            pred_y_valid = best_clf.predict_proba(X_valid)[:, 1]
            pred_y_test = best_clf.predict_proba(X_test)[:, 1]
        else:
            pred_y_valid = best_clf.predict(X_valid)
            pred_y_test = best_clf.predict(X_test)

        # add a col with the above predicted values (for each of the classifier)
        dataset_valid['predicted_label_'+clf] = pred_y_valid
        dataset_test['predicted_label_'+clf] = pred_y_test

        selected_cols.append('predicted_label_' + clf)

        # score on the validation dataset added in results.txt
        f.write('score on valid data for ' + clf + '=' + str(roc_auc_score(y_valid, pred_y_valid)) + '\n')
        print("valid roc score for ",clf,str(roc_auc_score(y_valid, pred_y_valid)))
        # score on the test dataset added in results.txt
        f.write('score on test data for ' + clf + '=' + str(roc_auc_score(y_test, pred_y_test)) + '\n')
        # print(roc_auc_score(y_test, pred_y_test))


    f.close()

    ## writing valid and test data, having columns:["pixel_id", "year", "Fire"] along with predicitons of all classifiers
    dataset_valid_selected_cols = dataset_valid[selected_cols]
    print('Writing predictions file on valid data')
    dataset_valid_selected_cols.to_csv(folder_saving + 'dataset_valid_with_predictions.csv')

    dataset_test_selected_cols = dataset_test[selected_cols]
    print('Writing predictions file on test data')
    dataset_test_selected_cols.to_csv(folder_saving + 'dataset_test_with_predictions.csv')

    # ROC scores
    # score on valid data for NeuralNet=0.8207489340163627
    # score on test data for NeuralNet=0.8319811839073478
    # score on valid data for XgBoost=0.8244967014266147
    # score on test data for XgBoost=0.8281527750829516

def get_the_best_classifier(folder_saving, classif_dict=None, tuned_parameters_dict=None):
    """
    Pipeline to get the best classifier/ best parameters of each of our classifiers.

    :param folder_saving: str, path to store the results.
    :param classif_dict: dictionary of classifiers. {name_classifier: classifier(), ...}, If None, automatic ones will
    be used.
    :param tuned_parameters_dict: dictionary of parameters to tune during grid search. If None, automatic ones
     will be used.
    """

    os.makedirs(folder_saving, exist_ok=True)

    print('1: loading train, test and valid sets.')
    input_df, column_list = read_data(csv_file)
    X_train, X_valid, X_test, y_train, y_valid, y_test = get_train_test_data(input_df)

    print('2: pre-processing features (standardization).') ##normalizing the features leads to better results
    X_train, X_valid, X_test = standardize_from_train(X_train, X_valid, X_test)

    print('3: build and compare classifiers')
    best_estimator_for_all_classifiers= \
        Sklearn_Methods.build_and_compare_classifiers_with_gridsearch(X_train, y_train, selected_features, folder_saving,
                                                                      classif_dict, tuned_parameters_dict)
    print('4: Saving results.')
    results_on_valid_test_dataset_with_gridsearch(best_estimator_for_all_classifiers, X_valid, y_valid, X_test, y_test,
                                             X_train, y_train, folder_saving)


if __name__=='__main__':

    folder_saving=path+"Results/"
    # current date
    now = datetime.datetime.now()
    date = now.strftime("%Y_%m_%d")
    date_with_hr_sec = now.strftime("%Y_%m_%d_%H_%M_%S")

    get_the_best_classifier(folder_saving+str(date_with_hr_sec)+"/")
















