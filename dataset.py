import torch.utils.data as data

from PIL import Image
import os
import os.path
import numpy as np
from numpy.random import randint

class VideoRecord(object):
    def __init__(self, row):
        self._data = row

    @property
    def path(self):
        return self._data[0]

    @property
    def num_frames(self):
        return int(self._data[1])

    @property
    def label(self):
        return int(self._data[2])


class RSDataSet(data.Dataset):
    def __init__(self, root_path, list_file,
                 num_segments=10,
                 image_tmpl='img_{:05d}.jpg', transform=None,
                 testing=False):

        self.root_path = root_path
        self.list_file = list_file
        self.num_segments = num_segments
        self.image_tmpl = image_tmpl
        self.transform = transform
        self.testing = testing

        self._parse_list()

    def _load_image(self, directory, idx):
        return [Image.open(os.path.join(directory, self.image_tmpl.format(idx))).convert('RGB')]

    def _parse_list(self):
        self.video_list = [VideoRecord(x.strip().split(' ')) for x in open(self.list_file) if VideoRecord(x.strip().split(' ')).num_frames>0]

    def _sample_indices(self, record):
        if record.num_frames < self.num_segments:
            indices = np.zeros(self.num_segments)
        else:
            interval = record.num_frames // self.num_segments
            indices = np.array([randint(interval*i, interval*(i+1)) for i in range(self.num_segments)])
        return indices + 1

    def __getitem__(self, index):
        record = self.video_list[index]
        segment_indices = self._sample_indices(record)
        return self.get(record, segment_indices)

    def get(self, record, indices):
        images = list()
        for seg_ind in indices:
            p = int(seg_ind)
            seg_imgs = self._load_image(record.path, p)
            images.extend(seg_imgs)
        process_data = self.transform(images)
        if self.testing:
            return process_data,record.path.split('/')[-1]
        else:
            return process_data,record.label,record.path.split('/')[-1]

    def __len__(self):
        return len(self.video_list)

