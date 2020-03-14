#!/bin/bash

if [[ ! -f "youtube_data.zip" ]]
then
  wget http://www.cs.put.poznan.pl/kmiazga/students/ped/youtube_data.zip
fi

if [[ ! -d "youtube_data" ]]
then
  unzip youtube_data.zip
fi
