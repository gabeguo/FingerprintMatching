for dir in "sd300a_split" "sd301_split_balanced" "sd302_split_balanced"
do
    echo $dir
    for modality in "train" "val" "test"
    do 
        echo "\t${modality}"
        for fgrp in 01 02 03 04 05 06 07 08 09 10
        do
            count=$(find "/data/therealgabeguo/fingerprint_data/${dir}/${modality}" -type f -name "*_${fgrp}.*" | wc -l)
            echo "\t\t${fgrp}: ${count}"
        done
        echo "\t\t~"
        num_people=$(ls "/data/therealgabeguo/fingerprint_data/${dir}/${modality}" | wc -l)
        echo "\t\tNum people: ${num_people}"
    done
    echo "---"
done