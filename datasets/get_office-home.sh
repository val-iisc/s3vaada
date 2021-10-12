#!/bin/bash

if [ ! -d "data" ]; then
    mkdir "data"
fi

if [ ! -d "data/Office-Home" ]; then
    mkdir "data/Office-Home"
fi


bash gdrive.sh "https://drive.google.com/uc?export=download&id=0B81rNlvomiwed0V1YUxQdC1uOTg" tmp.zip  
unzip tmp.zip -d data/Office-Home
rm tmp.zip