#!/bin/bash

export PYTHONPATH="$PWD/src"
python src/inference/run.py -i examples -c examples/cache -o examples/out