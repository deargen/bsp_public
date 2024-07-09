#!/bin/bash

export PYTHONPATH="$PWD/src"
python src/inference/prepare_data.py -i examples -c cache