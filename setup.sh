#!/bin/bash

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Set up virtual env and install dependencies
cd ${DIR}
virtualenv -p python3 env
source ${DIR}/env/bin/activate
pip install -r ${DIR}/requirements.txt

