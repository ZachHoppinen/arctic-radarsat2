secret = $1
uncompress.sh $1
georeference.sh
python rename_tifs.py
