#!/bin/bash
# Schedule execution of many runs
# Run from root folder with: bash scripts/schedule.sh
# python src/train.py experiment=sr16x

devices="[1,3,4,5]"

scale=4
echo "python src/train.py experiment=tsr${scale}x"
python src/train.py experiment=tsr${scale}x trainer.devices=${devices}

for scale in 2
do
    echo "Running scale $scale"
    echo "python src/train.py experiment=sr${scale}x"
    python src/train.py experiment=sr${scale}x trainer.devices=${devices}
    echo "python src/train.py experiment=tsr${scale}x"
    python src/train.py experiment=tsr${scale}x trainer.devices=${devices}

done