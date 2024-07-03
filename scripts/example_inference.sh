#!/bin/bash

export PYTHONPATH="$PWD/src"
python src/inference/run.py -i examples -c cache -o out