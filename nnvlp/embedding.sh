#!/usr/bin/env bash
dir="embedding"
if [ ! -d "$dir" ]; then
    mkdir "embedding"
fi
file="embedding/vectors.npy"
if [ -f "$file" ]; then
	echo "$file found."
else
	url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9WU93NEI1bGhmYmc"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
file="embedding/words.pl"
if [ -f "$file" ]; then
    echo "$file found."
else
    url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9SC1mRXpkbWhfUDA"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
file="embedding/unknown.npy"
if [ -f "$file" ]; then
    echo "$file found."
else
    url="https://drive.google.com/uc?export=download&id=0BytHkPDTyLo9VVlld1VlVVVoSHM"
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}" -O tmp
    c=`grep -o "confirm=...." tmp`
    wget --load-cookie cookie.txt --save-cookie cookie.txt "${url}&$c" -O "${file}"
    rm cookie.txt tmp
fi
