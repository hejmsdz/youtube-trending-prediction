#!/bin/bash

set -e

if [[ ! -f "youtube_data.zip" ]]
then
  wget http://www.cs.put.poznan.pl/kmiazga/students/ped/youtube_data.zip
fi

if [[ ! -d "youtube_data" ]]
then
  unzip youtube_data.zip
fi

if [[ ! -f "youtube_data/haarcascade_frontalface_default.xml" ]]
then
  wget -P youtube_data https://raw.githubusercontent.com/opencv/opencv/master/data/haarcascades/haarcascade_frontalface_default.xml
fi

if [[ ! -f "youtube_data/frozen_east_text_detection.pb" ]]
then
  wget -P youtube_data https://raw.githubusercontent.com/ZER-0-NE/EAST-Detector-for-text-detection-using-OpenCV/master/frozen_east_text_detection.pb
fi

if [[ ! -f "youtube_data/model_v6_23.hdf5" ]]
then
  wget -P youtube_data https://raw.githubusercontent.com/priya-dwivedi/face_and_emotion_detection/master/emotion_detector_models/model_v6_23.hdf5
fi

if [ ! -d ".venv" ]
then
  python3 -m venv .venv
fi

. .venv/bin/activate

if [ -f ".env" ]
then
  set -o allexport
  . ./.env
  set +o allexport
fi

missing_requirements=$(pip freeze --local | diff requirements.txt - | grep '^<' | wc -l)
if [ $missing_requirements -gt 0 ]
then
  pip install -r requirements.txt
fi

python3 -m jupyter notebook --notebook-dir=notebooks
