#!/bin/bash

for vs in 64 128 256 512 1024 4096 8192 16384 65536 524288 2097152 8388608 67108864 134217728 268435456
do
	python buffer.py --variable_size=$vs --batch_size=10 >> results
done
