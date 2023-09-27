#!/bin/bash

# see https://sartorius-research.github.io/LIVECell/


curl -H "GET /?list-type=2 HTTP/1.1" \
     -H "Host: livecell-dataset.s3.eu-central-1.amazonaws.com" \
     -H "Date: 20161025T124500Z" \
     -H "Content-Type: text/plain" http://livecell-dataset.s3.eu-central-1.amazonaws.com/ > files.xml

grep -oPm1 "(?<=<Key>)[^<]+" files.xml | sed -e 's/^/http:\/\/livecell-dataset.s3.eu-central-1.amazonaws.com\//' > urls.txt

mkdir -p LIVECell_dataset_2021/{annotations,models,nuclear_count_benchmark}


cat urls.txt | grep -E "image|annotation|nuclear_count_benchmark" > urls2.txt
mv urls2.txt urls.txt

while IFS="" read -r url || [ -n "$url" ]; do
  FILE="${url#"http://livecell-dataset.s3.eu-central-1.amazonaws.com/"}"
  wget "$url" -nc -O "$FILE"
  unzip -n "$FILE" -d "$(dirname "$FILE")"
done < urls.txt

rm files.xml
rm urls.txt
