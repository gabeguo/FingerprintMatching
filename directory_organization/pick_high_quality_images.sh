#!/bin/bash
postfix=".txt"
postfix_len=${#postfix}
directory_start="/data/verifiedanivray/nfiq_output/sd302_split/"
for file in $(find $directory_start -type f -name "*.txt" -print)
do
    quality=$(cat $file)
    if [ $quality -le 2 ]
    then
        #echo "$file"
        file_len=${#file}
        prefix_len=$(($file_len-$postfix_len))
        file_prefix=${file:0:prefix_len}
        file_subpath=${file_prefix#*$directory_start}
        new_filepath="/data/therealgabeguo/fingerprint_data/sd302_high_quality/train/${file_subpath}.png"
        new_dirname=$(dirname $new_filepath)
        filename=$(basename $new_filepath)
        pid=$(basename $new_dirname)
        old_filepath="/data/verifiedanivray/mindtct_output/sd302_split/${pid}/${filename}"
        #echo $old_filepath
        #echo $new_filepath
        #echo $new_dirname
        mkdir -p $new_dirname
        cp $old_filepath $new_filepath
        echo "copied $old_filepath to $new_filepath"
    fi
done