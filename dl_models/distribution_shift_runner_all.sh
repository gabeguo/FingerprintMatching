#!/bin/bash
source ../venv/bin/activate

#finger_array=(1 2 3 4 5)
#prior_array=(0.50 0.25 0.10 0.075 0.050 0.025 0.01 0.0075 0.0050 0.0025 0.001)
finger_array=(1 2 3)
prior_array=(0.50 0.25 0.10 0.025 0.01 0.0025 0.001)
cuda=$1

for finger in ${finger_array[@]}; do
    for prior in ${prior_array[@]}; do
        echo "Prior: "$prior
        bc_out=$(echo "1.0 / (2.0 * $prior)" | bc -l )
        scale_factor=$(python -c "import math; print(math.ceil($bc_out))")
        if (( $scale_factor>20 )); then
            scale_factor=20
        fi
        echo "Scale Factor: "$scale_factor
        #results_filename=$finger"to"$finger"_results.json"
        results_filename="geometric_analysis_results_1.json"
        python distribution_shift_tester_batched.py -d /data/therealgabeguo/fingerprint_data/sd302_split_balanced -w /data/therealgabeguo/embedding_net_weights.pth -p $prior -c $cuda -n $finger -o /data/verifiedanivray/results -s $scale_factor -a 0.95 -dfs -dws -dss -sss -l $results_filename
    done
done
