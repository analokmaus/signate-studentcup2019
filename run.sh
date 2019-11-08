#!/bin/bash

inference=""
while getopts dlh opt
do
  case $opt in
    d ) skipdata=true ;;
    l ) inference="--inference" ;;
    h ) usage ;;
    \? ) usage ;;
  esac
done

if [ "$skipdata" = true ]; then
  echo "skipping data preprocess"
else
  python preprocess/preprocess.py
  python preprocess/generate_feature.py
fi

python train/interpolate.py
python train/train_cat_v56.py $inference
python train/train_cat_v58.py $inference
python train/train_cat_v59.py $inference
python predict/predict.py
