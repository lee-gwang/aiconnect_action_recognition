import os

for i in range(1, 5):
    os.system(f'tools/dist_train.sh configs/recognition/tsm/gh_tsm_fold{i}_resize.py 4 --work-dir work_dirs/gh_tsm_resize_fold{i} --validate --seed 0 --deterministic')