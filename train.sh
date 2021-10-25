GPUS=$1

cd mmaction2
# train classifier model
bash tools/dist_train.sh configs/recognition/tsm/super_tsm_r50_modify.py $GPUS --work-dir data/results/model1 --validate --seed 0 --deterministic
bash tools/dist_train.sh configs/recognition/tsm/super_tsm_r50_modify_oversampling.py $GPUS --work-dir data/results/model2 --validate --seed 0 --deterministic

cd ..
cd mmdetection
# train detection model
bash tools/dist_train.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py $GPUS --work-dir data/results/model3 --seed 0 --deterministic

