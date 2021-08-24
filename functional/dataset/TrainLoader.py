import os
from PIL import Image
import cv2
import csv
import numpy as np
import random
import torch
import torch.utils.data as data
import torch
import torchvision.transforms as transforms

import functional.utils.custom_transforms as tr

cv2.setNumThreads(0)
ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def train_dataloader(csv_path="ytvos.csv", ref_num=1):
    if not csv_path.endswith(".csv"):
        print("Did not detect .csv file, scan dir {} and generate ytvos.csv".format(csv_path))
        ld = os.listdir(csv_path)
        with open(os.path.join(ROOT_DIR, 'ytvos.csv'), 'w') as f:
            filewriter = csv.writer(f)
            for l in ld:
                files = os.listdir(os.path.join(csv_path, l))
                n = len(files)
                files.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
                init = int(files[0].split('.')[0])
                filewriter.writerow([l, init, n])
        csv_path = os.path.join(ROOT_DIR, 'ytvos.csv')

    filenames = open(csv_path).readlines()
    print("SELECTED VIDEO NUMBER IS {}".format(len(filenames)))
    videos = [filename.split(',')[0].strip() for filename in filenames]
    start_frames = [int(filename.split(',')[1].strip()) for filename in filenames]
    num_frames = [int(filename.split(',')[2].strip()) for filename in filenames]

    all_index = np.arange(len(videos))
    np.random.shuffle(all_index)

    train_data = []
    total_num = ref_num + 1
    for index in all_index:
        frame_interval = np.random.choice([2, 5, 8], p=[0.4, 0.4, 0.2])

        image_pairs = []
        n_frames = num_frames[index]
        start_frame = start_frames[index]
        frame_indices = np.arange(start_frame, start_frame+n_frames, frame_interval)
        total_batch, batch_mod = divmod(len(frame_indices), total_num)
        if batch_mod > 0:
            frame_indices = frame_indices[:-batch_mod]
        frame_indices_batches = np.split(frame_indices, total_batch)
        for batches in frame_indices_batches:
            image_pair = [os.path.join(videos[index], '{:05d}.jpg'.format(frame))
                          for frame in list(batches)]
            image_pairs.append(image_pair)
        train_data.extend(image_pairs)
    print("SELECTED FRAME PAIR NUMBER IS {}".format(len(train_data)))
    return train_data


def image_loader(path):
    image = cv2.imread(path)
    image = np.float32(image)  #/ 255.0
    return image


class train_image_folder(data.Dataset):
    def __init__(self, root_path, filenames, args, training):
        self.args = args
        self.refs = filenames
        self.root_path = root_path
        self.composed_transforms = transforms.Compose([
            tr.Resize(self.args.img_size),
            tr.DivNorm(255.0),
            tr.ConvertToLAB(),
            tr.ToTensor(),
            tr.Normalize([50, 0, 0], [50, 127, 127])])

    def __getitem__(self, index):
        images_path = self.refs[index]
        video_name = images_path[0].split('/')[0]
        images = [image_loader(os.path.join(self.root_path, image)) for image in images_path]
        lab_images = self.composed_transforms(images)

        of_images = 1
        meta = {"video_name": video_name}
        return lab_images, of_images, meta

    def __len__(self):
        return len(self.refs)

