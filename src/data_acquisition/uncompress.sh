#!/bin/bash

secret=$1

if [ -z ${secret} ]; 
then echo "password is unset"; exit 1;
else echo "password is set"; 
fi

OLDIFS=$IFS

IN_DIR='/data/nga/raw/*.zip'
OUT_DIR='/data/nga/'

ZIPS=$(find $IN_DIR -name "*.zip")

for ZIP in /data/nga/raw/*.zip; do
    [ -e "$ZIP" ] || continue
    IFS="_" arr=($ZIP)
    IFS=" "
    pass="${arr[1]}_${arr[2]}_${arr[3]}_$1"
    7z -p$pass -o/data/nga x $ZIP
done

IFS=$OLDIFS
