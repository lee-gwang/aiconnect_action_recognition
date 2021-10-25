# install mmaction2
cd mmaction2
pip install -r requirements/build.txt
pip install -v -e .
ln -s ../data/ ./data

cd ..
# install mmdetection
pip install mmdet
cd mmdetection
pip install -r requirements/build.txt
pip install -v -e .
ln -s ../data/ ./data

# for instaboost
pip install instaboostfast
# for panoptic segmentation
pip install git+https://github.com/cocodataset/panopticapi.git
# for LVIS dataset
pip install git+https://github.com/lvis-dataset/lvis-api.git
# for albumentations
pip install albumentations>=0.3.2 --no-binary imgaug,albumentations

# else
pip install tqdm imgaug pandas

