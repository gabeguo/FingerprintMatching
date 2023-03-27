for dir in "sd300a_split" "sd300a_split_balanced" "sd301_split_balanced" "sd302_split_balanced" "mindtct_minutiae/sd302_split"
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
        total_num_samples=$(find "/data/therealgabeguo/fingerprint_data/${dir}/${modality}" -type f | wc -l)
        echo "\t\tNum people: ${num_people}"
        echo "\t\tNum samples: ${total_num_samples}"
    done
    echo "---"
done