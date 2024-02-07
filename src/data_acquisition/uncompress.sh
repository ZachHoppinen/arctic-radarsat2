#!/bin/bash

secret=$1

if [ -z ${secret} ]; 
then echo "password is unset"; exit 1;
else echo "password is set"; 
fi

OLDIFS=$IFS

IN_DIR='/data/GS-L/NGA/*.zip'
OUT_DIR='/data/nga/'

if test -f /data/GS-L/NGA/; then
  echo "File exists."
fi

ZIPS=$(find $IN_DIR -name "*.zip")
# echo $ZIPS

for ZIP in /data/GS-L/NGA/*.zip; do
    echo $ZIP
    [ -e "$ZIP" ] || continue
    IFS="_" arr=($ZIP)
    IFS=" "
    pass="${arr[1]}_${arr[2]}_${arr[3]}_$1"
    7z -p$pass -o/data/nga x $ZIP
done

IFS=$OLDIFS
