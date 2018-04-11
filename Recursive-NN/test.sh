#!/bin/bash

# verbose
set -x


infile=${INFILE:-models/RNN2_wvecDim_50_middleDim_30_step_1e-1_2.bin} # the pickled neural network
model=${MODEL:-RNN2} # the neural network type

# test the model on test data
python runNNet.py --inFile $infile --test --data "yeh_to_kamaal_ho_gaya" --model $model

# test the model on dev data
python runNNet.py --inFile $infile --test --data "dev" --model $model

# test the model on training data
python runNNet.py --inFile $infile --test --data "train" --model $model












