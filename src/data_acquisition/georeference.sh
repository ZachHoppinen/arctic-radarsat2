IN_DIR='/data/nga/'

for F in /data/nga/*/imagery_HH.tif; do
    out="$(dirname "$F")/georeferenced.tif"
    # echo $out
    gdalwarp -t_srs 'EPSG:4326' $F $out
done