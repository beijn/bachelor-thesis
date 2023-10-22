#!/usr/bin/bash

echo "#  ===  Deleting cache and computing embeddings anew."
rm -rf ~/.cache/micro-sam/TEMPORARY-TIMING || mkdir ~/.cache/micro-sam/TEMPORARY-TIMING

echo "#  ===  Embedding & segmenting on GPU."
job1=$(sbatch --parsable bin/run micro-sam/time.infer-amg.py)

echo "#  ===  Segmenting on GPU with CACHED embeddings."
job2a=$(sbatch --parsable --dependency=afterok:"$job1" bin/run micro-sam/time.infer-amg.py)

echo "#  ===  Segmenting on CPU with CACHED embeddings."
job2b=$(sbatch --parsable --dependency=afterok:"$job1" bin/run-cpu micro-sam/time.infer-amg.py cpu)
