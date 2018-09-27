#!/bin/bash

input_file=$1
output_file=${input_file%.*}.out

gcc -g $input_file pingpong.c -o $output_file -I../ -libverbs
