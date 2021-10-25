#!/usr/bin/env bash

cd ../
python build_rawframes.py /home/leegwang/comp/aiconnect_action/data/test/ ../../data/ava/rawframes/ --task both --level 1 --flow-type tvl1 --mixed-ext
echo "Raw frames (RGB and Flow) Generated"
cd ava/
