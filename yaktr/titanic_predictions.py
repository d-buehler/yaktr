#!/usr/local/bin/python3

import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from imputer import BasicImputer, AdvancedImputer
from optparse import OptionParser

def import_dataset(dataset_name, imputer_type='basic'):
	'''
	Loads a dataset from ../data, given the dataset filename.
	Returns a pandas DataFrame.
	:param dataset_name: the filename of the dataset
	'''

	# Dropping Ticket initially because there's no obvious info there (maybe worth checking for patterns?)
	# Might be worth dropping Fare bc expect that to be colinear with Pclass and/or Cabin... should definitely
	#	do _something_ about those variables
	# I don't think Embarked will be of much use either, would expect the value of that variable to be
	# 	available via the other kept variables. I'll keep for now though
	cols_to_keep = ['Survived', 'Pclass', 'Sex', 'Age', 'SibSp', 'Parch', 'Fare', 'Cabin', 'Embarked']
	if dataset_name == 'test.csv':
		cols_to_keep.remove('Survived')


	df = pd.DataFrame.from_csv("../data/%s" % dataset_name)[cols_to_keep]
	
	# Don't do this. Need to impute missing values somehow, can't drop
	# 	rows from the test dataset.
	#df = df.dropna()

	# Impute missing values
	if imputer_type == 'advanced':
		df = AdvancedImputer().fit_transform(df)
	else:
		df = BasicImputer().fit_transform(df)

	return df

def import_answers():
	'''
	Returns a dataframe with the true survived values for passengers in the test dataset
	'''
	answers_df = pd.DataFrame.from_csv("../data/gender_submission.csv")
	return answers_df

# TODO: More feature engineering
def prep_training_data(df, debug=False):
	'''
	Returns a dataframe containing the fully transformed training dataset
	:param df: raw training dataset, as a pandas DataFrame
	'''
	# Might need to so some extra manipulation here so that Pclass gets properly dummy-ified,
	#	pd.get_dummies(df) by default treats this column as numerical, while it's reeeeally
	#	categorical, so it doesn't get dummies created for it
	final_train_df = pd.get_dummies(df, drop_first=True)
	
	if debug==True:
		print ("Number of training dataset columns: {0}".format(len(final_train_df.columns)))
		print ("Number of rows in original training dataset: {0}".format(len(df)))
		print ("Number of rows in final training dataset: {0}".format(len(final_train_df)))

	return final_train_df

def prep_test_data(df, training_features, debug=False):
	'''
	Returns a dataframe containing the fully transformed test dataset
	:param df: the raw test dataset in a pandas DataFrame
	:param training_features: list containing the features in the training dataset
	'''
	# Might need to so some extra manipulation here so that Pclass gets properly dummy-ified,
	#	pd.get_dummies(df) by default treats this column as numerical, while it's reeeeally
	#	categorical, so it doesn't get dummies created for it
	final_test_df = pd.get_dummies(df)

	if debug == True:
		print ("Original test df len: {0}".format(len(df)))
		print ("Dummy-ified test df len: {0}".format(len(final_test_df)))
	

	# When using pd.get_dummies() on the test dataset, you need to
	#	deal with the fact that you might not have observations for
	#	every value of every field in the training set. Ex: Maybe
	#	a field in the training dataset contained 3 unique values, 
	# 	but only 2 unique value are in the test dataset - gotta
	#	create 0'ed out fields so the model can estimate properly
	for feature in training_features:
		if (feature not in final_test_df.columns):
			final_test_df[feature] = 0
	

	# You also might find that there are some values in the test dataset
	#	that do not appear in the training dataframe, which, when using
	#	pd.get_dummies(), would create features in the test dataset
	#	that are not in the model. Get rid of these!
	test_features = final_test_df.columns
	for feature in test_features:
		if (feature not in training_features):
			#if debug == True: # Kinda overkill
			#	print ("Dropping {0}".format(feature))
			final_test_df = final_test_df.drop(feature, axis=1)

	# Ensure the same column order
	final_test_df = final_test_df[training_features]

	if debug == True:
		print ("Number of columns in the final test dataframe: {0}".format(len(final_test_df.columns)))

	return final_test_df

def df_to_arrays(df, y_col_name):
	'''
	Takes a pandas DataFrame with dummy-ified features as input
		and returns X and Y numpy arrays
	:param df: a pandas DataFrame with features and y values
	'''
	feature_col_names = list(df.columns)
	feature_col_names.remove(y_col_name)
	
	x_arr = df[feature_col_names].values
	y_arr = df[y_col_name].values
	
	return x_arr, y_arr


def load_training_data(debug=False, imputer_type='basic'):
	'''
	Loads the raw training dataset and transforms it. Returns the transformed training dataset.
	'''
	print("Using {} imputation method".format(imputer_type))
	raw_train_df = import_dataset('train.csv', imputer_type=imputer_type)
	train_df = prep_training_data(raw_train_df, debug=debug)
	return train_df

def load_test_data(training_features, debug=False, imputer_type='basic'):
	'''
	Loads and transforms the test dataset. Returns the transformed test dataset.
	:param training_features: list of the features in the final training dataset, used
		to determine which features to include in the final test dataset
	:param debug: True -> print more stuff
	'''
	raw_test_df = import_dataset('test.csv', imputer_type=imputer_type)
	test_df = prep_test_data(raw_test_df, training_features=training_features, debug=debug)

	if debug == True:
		print ("Number of rows in raw test df: {0}".format(len(raw_test_df)))
		print ("Number of rows in final test df: {0}".format(len(test_df)))
		print ("Number of columns in final test df: {0}".format(len(test_df.columns)))

	return test_df

#TODO: allow for more types of classifiers to be trained
def train_classifier(X, Y, type='GBC'):
	'''
	Returns a sklearn classifier, just implementing GBC at first,
		but could add more later to test/compare other methods
	:param X: numpy array of features
	:param Y: numpy array of the dependent variable
	:param type: type of classifier to return, default is GradientBoostingClassifier (GBC)
	'''
	
	if type == 'GBC':
		est = GradientBoostingClassifier()
	
	est.fit(X, Y)
	
	return est

def predict_survival(classifier, test_data, prediction_col_name, debug=False):
	'''
	Uses the provided classifier and test dataset to predict survival
		for passengers in the test dataset
	:param classifier: a sklearn classifier
	:param test_data: the formatted test dataframe
	:param debug: True -> more printed stuff
	'''

	if debug == True:
		print ("Probability predicion for the first test row: {0}"
			.format(classifier.predict_proba(test_data[0:1].values)))
		print ("Class predicion for the first test row: {0}"
			.format(classifier.predict(test_data[0:1].values)))
	test_data[prediction_col_name] = classifier.predict(test_df.values)
	

	return test_data

# I think this might not be comparing to an actual answers dataset... gender_submission.csv might
# 	just be Kaggle's sample output from a really simple model... leaving as-is for now but should
#	change/remove this later, maybe just produce the submission file
def evaluate_predictions(test_df_with_predictions, y_hat, produce_file_for_kaggle=False):
	'''
	Determines and prints the number and percentage of correct predictions
		in the provided test dataset
	:param test_df_with_predictions: test dataset with a column containing the predictions
	:param y_hat: prediction of survival (1 or 0)
	'''
	answers_df = import_answers()
	nr_test_passengers = len(answers_df)
	nr_correct = 0
	nr_incorrect = 0
	for pid, _ in answers_df.iterrows():
		if test_df_with_predictions.loc[pid][y_hat] == answers_df.loc[pid]['Survived']:
			nr_correct += 1
		else:
			nr_incorrect += 1

	pct_correct = (nr_correct / nr_test_passengers) * 100.0
	print ("Correct: {0}\nIncorrect: {1}\nPercent correct: {2:.2f}%"
		.format(nr_correct, nr_incorrect, pct_correct))

	if produce_file_for_kaggle:
		produce_kaggle_file(test_df_with_predictions=test_df_with_predictions, y_hat=y_hat)

def produce_kaggle_file(test_df_with_predictions, y_hat):
	'''
	Produces a file named kaggle_submission.csv to submit to Kaggle's Titanic competition
	:param test_df_with_predictions: test dataset with a column containing the predictions
	'''
	header = "PassengerId,Survived\n"
	with open("kaggle_submission.csv", "w") as f:
		f.write(header)
		for pid, _ in test_df_with_predictions.iterrows():
			line = '{passenger_id},{survival_prediction:.0f}\n'.format(passenger_id=pid, survival_prediction=test_df_with_predictions.loc[pid][y_hat])
			f.write(line)

if __name__ == '__main__':
	
	# Parse command line args
	parser = OptionParser()
	parser.add_option("--debug", 
		dest='debug_mode', 
		action="store_true",
		default=False,
		help="Use this option to print more messages as the script runs")
	parser.add_option("--produce-submission-file",
		dest='produce_file',
		action="store_true",
		default=False,
		help="Use this option to produce a submission file for Kaggle. Default is False.")
	parser.add_option("--imputer-type",
		dest='imputer_type',
		default='basic',
		help='Type of imputation to use for training and test datasets')
	(options, args) = parser.parse_args()

	debug = options.debug_mode
	produce_submission_file = options.produce_file
	imputer_type = options.imputer_type

	y_name = 'Survived'
	train_df = load_training_data(debug=debug, imputer_type=imputer_type)
	X, Y = df_to_arrays(train_df, y_col_name=y_name)

	training_features = [ col for col in train_df.columns if col != 'Survived' ]
	test_df = load_test_data(training_features, debug=debug, imputer_type=imputer_type)

	# GBC
	print ("\n\nClassifier: GBC")
	y_hat_name = 'prediction_GBC'
	classifier = train_classifier(X, Y, type='GBC')
	test_df = predict_survival(classifier=classifier, test_data=test_df, prediction_col_name=y_hat_name ,debug=debug)
	evaluate_predictions(test_df_with_predictions=test_df, y_hat=y_hat_name, produce_file_for_kaggle=produce_submission_file)