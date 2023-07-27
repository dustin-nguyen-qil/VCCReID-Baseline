import argparse
import numpy as np
import random
import os.path as osp
import pickle
import os

parser = argparse.ArgumentParser(description='Dataset Preparation')
parser.add_argument('--root', type=str, required=True, help='path to data')
parser.add_argument('--dataset_name', type=str, default='vccr', required=True, help='vccr or ccvid or ccpg')

args = parser.parse_args()

root = args.root
dataset_name = args.dataset_name



def prepare(root, dataset_name): 
    data_path = osp.join('data', dataset_name)
    if osp.exists(data_path):
        pass
    else:
        os.mkdir(data_path)
    if dataset_name == 'ccvid':
        
        root = osp.join(root, 'CCVID')
        prepare_ccvid(root=root)
    elif dataset_name == 'vccr':
        root = osp.join(root, 'VCCR')
        prepare_vccr(root=root)

def prepare_vccr(root):
    train_dir = osp.join(root, 'train')
    test_dir = osp.join(root, 'test_qg')

    # Training data
    ids = os.listdir(train_dir)
    train_set = []
    for id in ids:
        id_path = osp.join(train_dir, id)
        tracklets = os.listdir(id_path)
        for tracklet in tracklets:
            tracklet_path = osp.join(id_path, tracklet)
            imgs = os.listdir(tracklet_path)
            random_img = random.choice(imgs)
            cam_id = random_img.split('-')[1][1:]
            clothes_id = random_img.split('-')[2][1:]
            img_paths = []
            for img in imgs:
                img_path = osp.join(tracklet_path, img)
                img_paths.append(img_path)
            img_paths = sorted(img_paths, key=lambda s: s.split('-')[-1][1:-4])
            train_set.append(
                {'p_id': int(id), 
                 'img_paths': img_paths,
                 'cam_id': int(cam_id),
                 'clothes_id': int(clothes_id)
                 }
            )
    train_content = {
        'data': train_set,
        'num_pids': int(len(ids)),
    }
    with open(osp.join('data/vccr', 'train.pkl'), 'wb') as f:
            pickle.dump(train_content, f)

    # query and gallery data
    query_gallery= {'query': [], 'gallery': []}

    for key in list(query_gallery.keys()):
        data_set = []
        dir = osp.join(test_dir, key)
        ids = os.listdir(dir)
        for id in ids:
            id_path = osp.join(dir, id)
            cams = os.listdir(id_path)
            for cam in cams:
                cam_path = osp.join(id_path, cam)
                tracklets = os.listdir(cam_path)
                for tracklet in tracklets:
                    tracklet_path = osp.join(cam_path, tracklet)
                    imgs = os.listdir(tracklet_path)
                    random_img = random.choice(imgs)
                    cam_id = random_img.split('-')[1][1:]
                    clothes_id = random_img.split('-')[2][1:]
                    if len(imgs) < 8:
                        continue
                    img_paths = []
                    for img in imgs:
                        img_path = osp.join(tracklet_path, img)
                        img_paths.append(img_path)
                    img_paths = sorted(img_paths, key=lambda s: s.split('-')[-1][1:-4])
                    data_set.append(
                        {'p_id': int(id), 'img_paths': img_paths, 'cam_id': int(cam_id), 'clothes_id': int(clothes_id)}
                    )
        query_gallery[key] = {
            'data': data_set,
            'num_pids': int(len(ids)),
        }
        with open(osp.join('data/vccr', f'{key}.pkl'), 'wb') as f:
            pickle.dump(query_gallery[key], f)

def prepare_ccvid(root):
    modes = ['train', 'query', 'gallery']
    for mode in modes:
        data_path = osp.join(root, f'{mode}.txt')
        tracklets, _, num_pids, _, num_clothes, _, _ = process_ccvid(root, data_path, relabel=True)
        data_set = []
        for item in tracklets:
            data_set.append({
                'img_paths': item[0],
                'p_id': item[1],
                'cam_id': item[2],
                'clothes_id': item[3],
            })
        content = {
            'data': data_set,
            'num_clothes': num_clothes,
            'num_pids': num_pids
        }
        with open(osp.join('data/ccvid', f'{mode}.pkl'), 'wb') as f:
            pickle.dump(content, f)


def process_ccvid(root, data_path, relabel=False, clothes2label=None):
    tracklet_path_list = []
    pid_container = set()
    clothes_container = set()
    with open(data_path, 'r') as f:
        for line in f:
            new_line = line.rstrip()
            tracklet_path, pid, clothes_label = new_line.split()
            tracklet_path_list.append((tracklet_path, pid, clothes_label))
            clothes = '{}_{}'.format(pid, clothes_label)
            pid_container.add(pid)
            clothes_container.add(clothes)
    pid_container = sorted(pid_container)
    clothes_container = sorted(clothes_container)
    pid2label = {pid:label for label, pid in enumerate(pid_container)}
    if clothes2label is None:
        clothes2label = {clothes:label for label, clothes in enumerate(clothes_container)}

    num_tracklets = len(tracklet_path_list)
    num_pids = len(pid_container)
    num_clothes = len(clothes_container)

    tracklets = []
    num_imgs_per_tracklet = []
    pid2clothes = np.zeros((num_pids, len(clothes2label)))

    for tracklet_path, pid, clothes_label in tracklet_path_list:
        tracklet_path = osp.join(root, tracklet_path)
        # img_paths = glob.glob(osp.join(root, tracklet_path, '*')) 
        img_paths = [osp.join(tracklet_path, img) for img in os.listdir(tracklet_path) if os.path.isfile(os.path.join(tracklet_path, img))]
        img_paths.sort()
        
        clothes = '{}_{}'.format(pid, clothes_label)
        clothes_id = clothes2label[clothes]
        pid2clothes[pid2label[pid], clothes_id] = 1
        if relabel:
            pid = pid2label[pid]
        else:
            pid = int(pid)
        session = tracklet_path.split('/')[0]
        cam = tracklet_path.split('_')[1]
        if session == 'session3':
            camid = int(cam) + 12
        else:
            camid = int(cam)

        num_imgs_per_tracklet.append(len(img_paths))
        tracklets.append((img_paths, pid, camid, clothes_id))

    num_tracklets = len(tracklets)

    return tracklets, num_tracklets, num_pids, num_imgs_per_tracklet, num_clothes, pid2clothes, clothes2label

if __name__ == "__main__":
    prepare(root, dataset_name)