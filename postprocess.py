import pandas as pd
import os
import numpy as np
from sklearn.metrics import accuracy_score, f1_score

class_dict = {'putup_umbrella': 0,
        'ride_kick': 1,
        'fall_down': 2,
        'driveway_walk': 3,
        'jay_walk': 4,
        'ride_moto': 5,
        'ride_cycle': 6,
        'fighting': 7,
        'normal': 8}

def func1(x, v, v2):
    value=v
    if x[0] == v2:
        if value in x.values:
            return value
        else:
            return x[0]
    else:
        return x[0]

def detect_postprocess():
    class_dict = {'putup_umbrella': 0,
        'ride_kick': 1,
        'fall_down': 2,
        'driveway_walk': 3,
        'jay_walk': 4,
        'ride_moto': 5,
        'ride_cycle': 6,
        'fighting': 7,
        'normal': 8}

    detect_results = np.load('./data/results/det_results.pkl', allow_pickle=True)

    # for frame matching
    img_path = []
    for (path, dir, files) in os.walk("./data/test_raw_frame/"):
        for filename in files:
            #ext = os.path.splitext(filename)[-1]
            if ('ipynb' not in path)&('.json' not in filename):
                img_path.append("%s/%s" % (path, filename))

    test = pd.DataFrame()            
    test['filename'] = img_path
    test['width'] = 960
    test['height'] = 540

    test['folder'] = test['filename'].apply(lambda x: x.split('test_raw_frame')[-1].split('/')[1])
    test = test.sort_values('folder').reset_index(drop=True)

    # detect confidence
    c1_num = []
    c2_num = []
    confidence = 0.9 # kick
    confidence2 = 0.5 # fall_down

    for i in detect_results:
        # class1
        len_ = i[0].shape[0]
        len_2 = i[1].shape[0]
        if len_!=0:
            c1_num.append((i[0][:,-1]>confidence).sum())
        else:
            c1_num.append(0)
        
        # class2    
        if len_2!=0:
            c2_num.append((i[1][:,-1]>confidence2).sum())
        else:
            c2_num.append(0)
        
        
    test['class1_num'] = c1_num # kick
    test['class2_num'] = c2_num # fall_down

    # aggregate
    dt = test.groupby('folder').sum().reset_index().copy()
    dt['folder'] = dt['folder'].apply(lambda x: x.split('_frames')[0] + '.mp4')
    dt = dt[dt['folder']!='test_video_0329.mp4'].reset_index(drop=True)


    return dt


def clf_postprocess():
    class_dict = {'putup_umbrella': 0,
        'ride_kick': 1,
        'fall_down': 2,
        'driveway_walk': 3,
        'jay_walk': 4,
        'ride_moto': 5,
        'ride_cycle': 6,
        'fighting': 7,
        'normal': 8}

    # ensemble results
    a=0
    a += np.array(np.load(f'./data/results/model1.pkl', allow_pickle=True))/2
    a += np.array(np.load(f'./data/results/model2.pkl', allow_pickle=True))/2
    
    dict_ = {class_dict[x]:x for x in class_dict}
    sub = pd.read_csv('./data/sample_submission.csv')

    # argsort results
    topk = 5
    b = a.argsort(1)[:,::-1][:,:topk]

    ss = pd.DataFrame()
    ss['f'] = sub['video_filename']
    ss['pred'] = a.argmax(1)
    ss['pred2'] = b[:,1]
    ss['pred3'] = b[:,2]
    ss['pred4'] = b[:,3]
    ss['pred5'] = b[:,4]

    ss[f'prob_0'] = a[:,0] # umbrella

    # unbrella postprocessing (If the probability value predicted by the umbrella is less than 0.95, select the second prediction value.)
    prob = 0.95 # 0.95 is better than 0.9 in lb score
    ss2 = ss[ss['pred']==0].copy()
    idx_ = ss[ss['pred']==0].index 
    ss2.loc[ss2[f'prob_0']<prob, 'pred'] = ss2[ss2[f'prob_0']<prob]['pred2']

    ss.loc[idx_, 'pred'] = ss2['pred']

    # ##############
    # 추가적인 post할시에 써보기
    col = ['post','pred2','pred3']
    ss['post'] = ss['pred']
    ss['post']= ss[col].apply(lambda x: func1(x, 3, 4) ,1) # 무단횡단을 차도보행 # 1
    ss['post']= ss[col].apply(lambda x: func1(x, 4, 8) ,1)# 보통을 무단횡단 # 2
    ss['post']= ss[col].apply(lambda x: func1(x, 3, 1) ,1) # 킥보드를 차도보행 # 3
    ss['post']= ss[col].apply(lambda x: func1(x, 3, 7) ,1) # 폭력을 차도보행 # 4
    
    return ss

if __name__ == "__main__":
    # postprocessing
    det = detect_postprocess()
    clf = clf_postprocess()
    det['post'] = clf['post']
    det['post2'] = clf['post']
    det.loc[((det['class1_num']>20)&(det['post'].isin([3,4]))), 'post2'] = 1
    det.loc[(det['class2_num']>10)&(det['post'].isin([3,4,8])), 'post2'] = 2 

    # submission
    sub = pd.read_csv('./data/sample_submission.csv')
    dict_ = {class_dict[x]:x for x in class_dict}
    sub['class'] = det['post2'].apply(lambda x: dict_[x]) # transform to action_name
    sub.to_csv('./data/results/final.csv', index=False)
    print('final submission saved!')


