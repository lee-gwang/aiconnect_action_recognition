GPUS=$1

# classifier
cd mmaction2
## model 1
bash tools/dist_test.sh configs/recognition/tsm/super_tsm_r50_modify.py data/pretrained/best_model.pth $GPUS --out data/results/model1.pkl
## model 2
bash tools/dist_test.sh configs/recognition/tsm/super_tsm_r50_modify.py data/pretrained/best_over_model.pth $GPUS --out data/results/model2.pkl

cd ..
cd mmdetection
# detector
bash tools/dist_test.sh configs/faster_rcnn/faster_rcnn_r50_fpn_1x_coco.py data/pretrained/det.pth $GPUS --out data/results/det_results.pkl

cd ..
# post processing
python postprocess.py

