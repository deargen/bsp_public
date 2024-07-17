# bsp_public

## Setup
```
docker pull daeseoklee/bsp-inference
```

## Model checkpoints
Model checkpoints (including those from the ablation study) are included in the docker image. Optionally, one can download them from [huggingface](https://huggingface.co/datasets/daeseoklee/bsp_data/tree/main). After downloading `train_logs.tar` and unzipping, one can perform inference by running `src/inference/run.py` without using docker. 

## Execution 
### Inference from pdb inputs  
```
# examples/ is a directory containing pdb files 
./predict-binding-site examples examples/out.csv
```

### Inference from dataset inputs
Optionally, one can perform inference for the datasets appearing in our paper (scPDB etc.). The datasets (preprocessed in HDF5 format) can be downloaded from [huggingface](https://huggingface.co/datasets/daeseoklee/bsp_data/tree/main). Then, one can run, for example, 
```
./predict-binding-site scPDB_cache_partitioned/scPDB_cache_1.hdf5 scPDB_out_1.csv 
```

### Replicating the case study results 
```
rm ./casestudy/out.csv
bash ./casestudy/inference.sh 
```