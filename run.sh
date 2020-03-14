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

if [ ! -d ".venv" ]
then
  python3 -m venv .venv
fi

. .venv/bin/activate

missing_requirements=$(pip freeze --local | diff requirements.txt - | grep '^<' | wc -l)
if [ $missing_requirements -gt 0 ]
then
  pip install -r requirements.txt
fi

python3 -m jupyter notebook --notebook-dir=notebooks
