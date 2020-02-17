#!/bin/bash

../argparser.sh "$@"
start=`date +%s`



end=`date +%s`
runtime=$((end-start))

echo "Completed in " $runtime " seconds"