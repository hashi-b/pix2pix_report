#!/bin/bash

export PATH=/opt/cuda-8.0/bin:$PATH
export LD_LIBRARY_PATH=/opt/cuda-8.0/lib64:$LD_LIBRARY_PATH
export PYTHONPATH=/home/hashib/dnnlib:$PYTHONPATH
export PYTHONPATH=/home/hashib/.local/lib:$PYTHONPATH

rootdir=./
outdir=./result/$1
mkdir -pv ${outdir}

python ./python/pix2pix.py -o ${outdir}


