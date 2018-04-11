#!/bin/bash

# verbose
set -x
###################
# Update items below for each train/test
###################

# training params
epochs=${NEPOCH:-30}
step=1e-1
wvecDim=${WVECDIM:-50}

# for RNN2 only, otherwise doesnt matter
middleDim=${MIDDLEDIM:-5}

model="RNN2"


########################################################
# Probably a good idea to let items below here be
########################################################

outfile="models/${model}_wvecDim_${wvecDim}_middleDim_${middleDim}_step_${step}_2.bin"

python -u runNNet.py --step $step --epochs $epochs --outFile $outfile \
                --middleDim $middleDim --outputDim 3 --wvecDim $wvecDim --model $model
