#!/bin/bash

echo TODO: make this dump into ~/.cache/thesis/data


URL='https://github.com/matterport/Mask_RCNN/releases/download/v2.1/balloon_dataset.zip'

DATADIR="$HOME/.cache/thesis/data"


wget "$URL" -nc -O "$DATADIR/balloon_dataset.zip"
unzip -n "$DATADIR/balloon_dataset.zip" -d "$DATADIR"
