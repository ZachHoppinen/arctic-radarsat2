IN_DIR='/data/nga/ncs'

for DIR in "$IN_DIR"/*; do
    echo $DIR
    check='First run'
    for F in "$DIR"/*.nc; do
        echo $F
        if [ "$check" ]; then
            first=$F
        else 
            echo "gdalwarp $F $first"
        fi
        check=''
    done
    # out="$(dirname "$F")/georeferenced.tif"
    # echo $F
    # gdalwarp -t_srs 'EPSG:4326' $F $out
done