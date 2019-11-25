from __future__ import print_function, division
import os
from skimage import io, transform
import json
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class VQADataset(Dataset):
    """vqa dataset."""

    def __init__(self, json_file, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        data = json.load(open(json_file))
        self.questions = pd.DataFrame(data["questions"])
        self.root_dir = root_dir

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    

        print(self.questions.head())
        print(self.questions.loc[0])
        image_id = self.questions.iloc[idx]["image_id"]
        image_id = f'abstract_v002_train2015_{image_id:012}.png'
        question_txt = self.questions.iloc[idx]["question"]
        img_name = os.path.join(self.root_dir, image_id)
        print(img_name)
        image = io.imread(img_name)
        sample = {'image': image, 'question': question_txt}

        return sample

train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
train_images = 'scene_img_abstract_v002_train2015'
vqa_dataset = VQADataset(json_file=train_json, root_dir=train_images)

fig = plt.figure()

for i in range(len(vqa_dataset)):
    sample = vqa_dataset[i]

    print(i, sample['image'].shape, sample['question'])

    if i == 3:
        plt.imshow(sample["image"])
        plt.show()
        break
