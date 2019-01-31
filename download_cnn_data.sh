#!/bin/bash
fileid="1VFKeAZZutQoFi-ARBJ8R0xXJUIdb23Ig"
filename="google_data.tgz"
curl -c ./cookie -s -L "https://drive.google.com/uc?export=download&id=${fileid}" > /dev/null
curl -Lb ./cookie "https://drive.google.com/uc?export=download&confirm=`awk '/download/ {print $NF}' ./cookie`&id=${fileid}" -o ${filename}

