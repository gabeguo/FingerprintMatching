#!/bin/bash
postfix="_minutiae.png"
directory_start="/data/verifiedanivray/mindtct_output/"
postfix_len=${#postfix}
for file in $(find /data/verifiedanivray/mindtct_output -type f -name "*_minutiae.png" -print)
do
    file_len=${#file}
    prefix_len=$(($file_len-$postfix_len))
    file_prefix=${file:0:prefix_len}
    file_subpath=${file_prefix#*$directory_start}
    new_filepath="/data/therealgabeguo/mindtct_minutiae/${file_subpath}.png"
    new_dirname=$(dirname $new_filepath)
    mkdir -p $new_dirname
    cp $file $new_filepath
done