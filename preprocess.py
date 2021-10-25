from sklearn.model_selection import StratifiedKFold, KFold
import pandas as pd
# for classifier data
def make_clf_data(oversample=False):

    class_dict = {'putup_umbrella': 0,
    'ride_kick': 1,
    'fall_down': 2,
    'driveway_walk': 3,
    'jay_walk': 4,
    'ride_moto': 5,
    'ride_cycle': 6,
    'fighting': 7,
    'normal': 8}

    try:
        train = pd.read_csv('./data/train_list_rawframes_fold0.txt', sep=' ')
        train2 = pd.read_csv('./data/train_list_rawframes_fold0_tw.txt', sep=' ')
        val = pd.read_csv('./data/val_list_rawframes_fold0.txt', sep=' ')
        test = pd.read_csv('./data/test_list_rawframes.txt', sep=' ')
        print('already exist dataset')


    except:    
        print('preprocess dataset...')
        df = pd.read_csv('./data/train.csv')
        df['label'] = df['class'].apply(lambda x: class_dict[x])
        df['frame_num'] = 63 # mistake

        skf = StratifiedKFold(n_splits=5, random_state=42, shuffle=True)
        df['fold'] = -1
        for n_fold, (_,v_idx) in enumerate(skf.split(df, df['label'])):
            df.loc[v_idx, 'fold']  = n_fold

        # to txt
        for i in range(5):
            train = df[df['fold']!=i].copy()
            val = df[df['fold']==i].copy()
            train['path'] =  train['video_filename'].apply(lambda x: x.split('.mp4')[0]) + '_frames'
            val['path'] =  val['video_filename'].apply(lambda x: x.split('.mp4')[0]) + '_frames'
            train = train[['path', 'frame_num', 'label']]
            val = val[['path', 'frame_num', 'label']]

            if oversample:
                tw = train[train['label'].isin([6, 1 ,7,0,4,2])]
                train = pd.concat([train, tw, tw]).reset_index(drop=True)
                train.to_csv(f'./data/train_list_rawframes_fold{i}_tw.txt', sep=' ', header=None, index=False)
            else:
                train.to_csv(f'./data/train_list_rawframes_fold{i}.txt', sep=' ', header=None, index=False)
                val.to_csv(f'./data/val_list_rawframes_fold{i}.txt', sep=' ', header=None, index=False)    

            break
            
            
        # test
        test = pd.read_csv('./data/sample_submission.csv')
        test['frame_num'] = 68 # mistake
        test['path'] =  test['video_filename'].apply(lambda x: x.split('.mp4')[0]) + '_frames'
        test['label'] = 0
        test = test[['path', 'frame_num', 'label']]
        test.to_csv(f'./data/test_list_rawframes.txt', sep=' ', header=None, index=False)

    return

# for detector dataset
def pd_to_coco(data, save_json_path, categories, label=True):
    column_names =['filename','categoryid','width', 'height','xmin','ymin','xmax','ymax'] # 안쓰임 참고용


    #path = 'train_label.csv' # the path to the CSV file
    

    images = []
    annotations = []

    data['fileid'] = data['filename'].astype('category').cat.codes
    data['annid'] = data.index

    def image(row):
        image = {}
        image["height"] = row.height
        image["width"] = row.width
        image["id"] = row.fileid
        image["file_name"] = row.filename
        return image

    def annotation(row):
        annotation = {}
        area = (row.xmax -row.xmin)*(row.ymax - row.ymin)
        annotation["segmentation"] = []
        annotation["iscrowd"] = 0
        annotation["area"] = area
        annotation["image_id"] = row.fileid

        annotation["bbox"] = [row.xmin, row.ymin, row.xmax -row.xmin,row.ymax-row.ymin ]

        annotation["category_id"] = row.categoryid
        annotation["id"] = row.annid
        return annotation
    
    if label:
        for row in data.itertuples():
            annotations.append(annotation(row))
    else:
        annotations=[]

    imagedf = data.drop_duplicates(subset=['fileid']).sort_values(by='fileid')
    for row in imagedf.itertuples():
        images.append(image(row))

    data_coco = {}
    data_coco["images"] = images
    data_coco["categories"] = categories # dictionary?
    data_coco["annotations"] = annotations
    json.dump(data_coco, open(save_json_path, "w"), indent=4)

if __name__ == "__main__":
    make_clf_data()
    make_clf_data(oversample=True)
