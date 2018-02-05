# yaktr
### Yet another Kaggle Titanic repo

## Setup
Setup requires [virtualenv](https://virtualenv.pypa.io/en/stable/) to be installed:
```bash
[your-shell]$ pip install virtualenv
```

Run setup.sh: Creates an environment (./env) and installs dependencies
```bash
[your-shell]$ setup.sh
```

## Data
Data from https://www.kaggle.com/c/titanic/data available in the data/ directory

## Usage
```bash
[~/Documents/github/yaktr]$ pwd
/Users/dbuehler720/Documents/github/yaktr
[~/Documents/github/yaktr]$ source env/bin/activate
(env) [~/Documents/github/yaktr]$ python titanic_predictions.py --help
usage: titanic_predictions.py [-h] [--debug]
                              [--output-filename OUTPUT_FILENAME]
                              [--imputer-type IMPUTER_TYPE]
                              [--comparison-set COMPARISON_SET]

optional arguments:
  -h, --help            show this help message and exit
  --debug               Use this option to print more messages as the script
                        runs
  --output-filename OUTPUT_FILENAME
                        Name for the filename containing the submission file
                        for Kaggle. Default is 'kaggle_submission.csv'.
  --imputer-type IMPUTER_TYPE
                        Type of imputation to use for training and test
                        datasets. Pass "basic" or "advanced", defaults to
                        basic if anything else is passed.
  --comparison-set COMPARISON_SET
                        Filename (fullpath) containing comparison set of
                        output to compare this runs set with
```

## Submissions
First attempt, Gradient boosting, minimal/almost no feature engineering: 0.77511

![Basic and Advanced scores](https://user-images.githubusercontent.com/7842856/35792150-ae70ad3a-0a19-11e8-9e4f-85d078abbf8a.jpg)
