#!/bin/bash
# see https://sartorius-research.github.io/LIVECell/


DATADIR="$HOME/.cache/thesis/data"

echo -e "\n\e[1;32m   >>=  Installing LIVECell data set into '$DATADIR'  =<<   \e[0m\n"


curl -H "GET /?list-type=2 HTTP/1.1" \
     -H "Host: livecell-dataset.s3.eu-central-1.amazonaws.com" \
     -H "Date: 20161025T124500Z" \
     -H "Content-Type: text/plain" http://livecell-dataset.s3.eu-central-1.amazonaws.com/ > files.xml

grep -oPm1 "(?<=<Key>)[^<]+" files.xml | sed -e 's/^/http:\/\/livecell-dataset.s3.eu-central-1.amazonaws.com\//' > urls.txt

cat urls.txt | grep -E "image|annotation|nuclear_count_benchmark" > urls2.txt
mv urls2.txt urls.txt


mkdir -p "$DATADIR"/DATA/LIVECell_dataset_2021/{annotations,models,nuclear_count_benchmark}


while IFS="" read -r url || [ -n "$url" ]; do
  FILE="$DATADIR/${url#"http://livecell-dataset.s3.eu-central-1.amazonaws.com/"}"
  wget "$url" -nc -O "$FILE"
  unzip -n "$FILE" -d "$(dirname "$FILE")"
done < urls.txt

rm files.xml
rm urls.txt
