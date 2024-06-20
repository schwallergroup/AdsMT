#!/bin/bash

[ ! -d "datasets" ] && mkdir -p datasets

# wget https://figshare.com/ndownloader/files/46807144 -O datasets/Alloy-GMAE.tar.gz
# wget https://figshare.com/ndownloader/files/46807147 -O datasets/FG-GMAE.tar.gz
# wget https://figshare.com/ndownloader/files/46807150 -O datasets/OCD-GMAE.tar.gz
# wget https://figshare.com/ndownloader/files/47125474 -O datasets/OC20-LMAE.tar.gz

cd datasets
wget https://zenodo.org/records/12104162/files/Alloy-GMAE.tar.gz
wget https://zenodo.org/records/12104162/files/FG-GMAE.tar.gz
wget https://zenodo.org/records/12104162/files/OCD-GMAE.tar.gz
wget https://zenodo.org/records/12104162/files/OC20-LMAE.tar.gz

tar zxf Alloy-GMAE.tar.gz
tar zxf FG-GMAE.tar.gz
tar zxf OCD-GMAE.tar.gz
tar zxf OC20-LMAE.tar.gz
rm *.tar.gz
cd ..
