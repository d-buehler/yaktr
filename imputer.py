
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import TransformerMixin

REGRESSION_COLUMNS = ['Age', 'Fare']
CLASSIFICATION_COLUMNS = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Cabin', 'Embarked']

# BasicImputer is shamelessly stolen from StackOverflow: https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
class BasicImputer(TransformerMixin):
    '''
    Shamelessly stolen from StackOverflow, renamed, and modified in other class(es?) below
    https://stackoverflow.com/questions/25239958/impute-categorical-missing-values-in-scikit-learn
    '''
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed with the most frequent value 
        in column.

        Columns of other types are imputed with mean of column.

        """
    def fit(self, X, y=None):

        self.fill = pd.Series([X[c].value_counts().index[0]
            if X[c].dtype == np.dtype('O') else X[c].mean() for c in X],
            index=X.columns)

        return self

    def transform(self, X, y=None):
        return X.fillna(self.fill)

# Created AdvancedImputer myself, uses the above BasicImputer as a basic template. _Slight_ improvement
#   on score vs. BasicImputer :-D
#   As called in titanic_predictions.py (using fit_transform()) - the fit() method is called first, 
#   then the transform() method.
class AdvancedImputer(TransformerMixin):
    def __init__(self):
        """Impute missing values.

        Columns of dtype object are imputed using
        a gradient boosting classifier to determine 
        the most likely value for missing values

        Columns of other types are imputed using a 
        gradient boosting regressor.
        """


    @staticmethod
    def transform_X_est(est_X_df, training_features):
        '''
        Dummy-ifies columns in the estimation dataset (dataset where values need to be imputed),
            adds columns if they exist in the dummy-ified training dataset but not the estimation
            datset, and removes columns if they exist in the estimation dataset but not the training
            datset.
        '''
        est_X_df_t = pd.get_dummies(est_X_df)
        for feature in training_features:
            if (feature not in est_X_df_t.columns):
                est_X_df_t[feature] = 0


        test_features = est_X_df_t.columns
        for feature in test_features:
            if (feature not in training_features):
                est_X_df_t = est_X_df_t.drop(feature, axis=1)
        # Ensure the same column order
        est_X_df_t = est_X_df_t[training_features]

        return est_X_df_t



    @staticmethod
    def transform_X_train (X_df):
        '''
        Dummy-ifies columns in the training dataset
        '''
        final_X_for_est = pd.get_dummies(X_df, drop_first=True)
        return final_X_for_est

    @staticmethod
    def transform_y_train(y, colname):
        '''
         Builds the map of the original y-names to y-codes and uses this map to encode the y values in
            the training dataset, only necessary for columns that will use a classifier
        '''
        if colname in CLASSIFICATION_COLUMNS:
            y_names = sorted(set(y))
            y_map = { y_name: y_code + 1 for y_code, y_name in enumerate(y_names) }
            print("y_map: {}".format(y_map))
            y_transformed = pd.Series([y_map[y_name] for y_name in y])
        else:
            y_map = None
            y_transformed = y

        return y_transformed, y_map

    def get_imputing_model(self, train_X, train_y, colname):
        '''
        Trains a RandomForestClassifier or a GradientBoostingRegressor to use for estimating missing values
            in a column. Which column is getting a model generated is determined when calling the function
            (train_Y = [the variable to create a model for imputing], train_X = [other independent variables])
        '''
        train_X_t = self.transform_X_train(train_X)
        train_y_t, y_map = self.transform_y_train(train_y, colname)
        training_features = train_X_t.columns

        train_X_t_arr = train_X_t.values
        train_y_t_arr = train_y_t.values

        model = GradientBoostingClassifier() if colname in CLASSIFICATION_COLUMNS else GradientBoostingRegressor()
        print("Predicting with {}".format(type(model)))

        model.fit(train_X_t_arr, train_y_t_arr)

        return model, training_features, y_map

    def get_predictions(self, model, X_est, training_features, y_map):
        '''
        Returns a pd.Series of predictions using the trained model and the subset
            of rows that require predictions for this column (meaning - the subset
            of rows where the column being predicted is null in the original dataset)
        '''
        X_est_t = self.transform_X_est(X_est, training_features)
        predictions_list = model.predict(X_est_t.values)

        # If a y_map is passed, use it to map the predictions back
        #    to their original values
        if y_map is not None:
            y_map_flipped = { v: k for k, v in  y_map.items() }
            print(y_map_flipped.keys())
            predictions = pd.Series([y_map_flipped[prediction] for prediction in predictions_list], index=X_est_t.index)
        else:
            predictions = pd.Series(predictions_list, index=X_est_t.index)

        return predictions

    def add_to_predictions_dict(self, predictions_series, col_name):
        '''
        Adds the predictions for [col_name] to the dict of predictions - to be
            used later to impute missing values into the original dataset
        '''
        try:
            print("Adding predictions for null values of {} to predictions_dict"
                .format(col_name))
            self.predictions_dict[col_name] = predictions_series
        except:
            print("No dict of predictions yet. Creating a new one.")
            self.predictions_dict = {}
            self.predictions_dict[col_name] = predictions_series


    def fit(self, X, y=None):
        '''
        Builds a dictionary of predictions for null values of each column in the X dataset,
            using sklearn classifiers/regression models (depending on if the data type being
            imputed is categorical or numerical, respectively).
            
            Dict is of the form... { col_name: pd.Series of predictions }
        '''
        
        # Make a BasicImpute-d version of X - pull rows from this 
        #    where col-to-impute is NOT NULL to create the model,
        #    and use the model to estimate the value of col-to-impute
        #    where it IS null in the original dataset
        self.X_basic_impute = X.copy()
        # Remove 'Survived' if it's in the dataset
        try:
            X_basic_impute.drop('Survived')
        except:
            print("No Survived column, moving on")
        self.X_basic_impute = BasicImputer().fit_transform(self.X_basic_impute)

        impute_columns = [ col for col in X.columns if col != 'Survived' ]
        for impute_column in impute_columns:
            print("Begin imputing {}".format(impute_column))

            # Get a dataframe of all rows where the column we want
            #    to impute is not null, and impute values for all
            #    other columns in _that_ dataframe.
            other_columns = [ c for c in X.columns if c is not impute_column ]
            populated_rows = list(X[X[impute_column].notnull()].index)
            rows_to_predict = list(X[X[impute_column].isnull()].index)
            print("Other columns: {0}\nNr populated: {1}\nNr to predict: {2}"
                .format(
                    other_columns,
                    len(populated_rows),
                    len(rows_to_predict)
                )
            )

            # If nothing to predict (no nulls), move on to the next column
            if len(rows_to_predict) == 0:
                print("Nothing to predict for {}, skipping".format(impute_column))
                self.add_to_predictions_dict(X[impute_column].copy(), impute_column)
                continue

            # Rows where the column is populated in the original dataset,
            #    Columns that are NOT the column we're estimating
            this_X = self.X_basic_impute[other_columns].loc[populated_rows].copy()
            this_y = X[impute_column].loc[populated_rows]

            model, training_features, y_map = self.get_imputing_model(train_X=this_X, train_y=this_y, colname=impute_column)

            # Only _need_ to predict y for values where it is null 
            #    in the original dataset
            est_y = self.get_predictions(model, 
                                    self.X_basic_impute[other_columns].loc[rows_to_predict], 
                                    training_features,
                                    y_map)

            self.add_to_predictions_dict(est_y, impute_column)

        if 'Survived' in X.columns:
            self.predictions_dict['Survived'] = X['Survived'].copy()

        return self

    def transform(self, X, y=None):
        '''
        Returns a dataframe containing filled-in missing values using predicted values
        '''
        # Similar to joining on the index, and creating a column
        #    like... coalesce(X.col, est_y.col)
        X_transformed = pd.DataFrame(columns=X.columns)
        for col in self.predictions_dict.keys():
            X_transformed[col] = X[col].combine_first(self.predictions_dict[col])
        return X_transformed
