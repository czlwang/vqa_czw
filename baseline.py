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

    def __init__(self, json_file, root_dir, annotation_json):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            annotation_json: contains the human answers
        """
        data = json.load(open(json_file))
        self.questions = pd.DataFrame(data["questions"])
        self.root_dir = root_dir
        anno_data = json.load(open(annotation_json))
        self.annotations = pd.DataFrame(anno_data["annotations"])

    def __len__(self):
        return len(self.questions)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
    

        question = self.questions.iloc[idx]
        image_id = question["image_id"]
        image_id = f'abstract_v002_train2015_{image_id:012}.png'
        question_txt = question["question"]
        question_id = question["question_id"]
        #print(self.annotations.head())
        answer_txt = self.annotations.loc[self.annotations["question_id"]==question_id].iloc[0]["multiple_choice_answer"]
        img_name = os.path.join(self.root_dir, image_id)
        print(img_name)
        print(self.questions.iloc[idx])
        image = io.imread(img_name)
        sample = {'image': image, 'question': question_txt, "answer": answer_txt}

        return sample

train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
train_images = 'scene_img_abstract_v002_train2015'
annotation_json = 'abstract_v002_train2015_annotations.json'
vqa_dataset = VQADataset(json_file=train_json, root_dir=train_images, annotation_json=annotation_json)

fig = plt.figure()

for i in range(len(vqa_dataset)):
    sample = vqa_dataset[i]

    print(i, sample['image'].shape, sample['question'], sample["answer"])

    if i == 3:
        plt.imshow(sample["image"])
        plt.show()
        break
