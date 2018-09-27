#/bin/bash

set -x

input_file=$1
output_file=${input_file%.*}.out

mpic++ -g $input_file -o $output_file -I ../Libraries/include -I $CUDA_PATH/include -L ../Libraries/lib -L $CUDA_PATH/lib64 -lcuda -lcudart -lmp -lgdsync -lgdrapi
