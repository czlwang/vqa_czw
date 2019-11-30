import torchvision.models as models
import cv2
import torchvision.transforms as transforms
import re
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
        annotations = pd.DataFrame(anno_data["annotations"])
        self.questions = self.questions.join(annotations.set_index('question_id'), on='question_id', rsuffix="_anno")

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
        answer_txt = question["multiple_choice_answer"]
        img_name = os.path.join(self.root_dir, image_id)
        image = cv2.imread(img_name)
        sample = {'image': image, 'question': question_txt, "answer": answer_txt}

        return sample

class Lang:
    def __init__(self, name):
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "SOS", 1: "EOS", 2:"UNK"}
        self.n_words = 3

    def addSentence(self, sentence):
        for word in sentence.split(' '):
            self.addWord(word)

    def addWord(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

def normalizeString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    print(s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, 2) for word in sentence.split(' ')]

def readQuestions(train_json):
    lang = Lang("questions")
    data = json.load(open(train_json))
    questions = pd.DataFrame(data["questions"])
    #for idx in range(len(questions)):
    for idx in range(10):
        question = questions.iloc[idx]
        lang.addSentence(normalizeString(question["question"]))
    return lang

train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
train_images = 'scene_img_abstract_v002_train2015'
annotation_json = 'abstract_v002_train2015_annotations.json'
vqa_dataset = VQADataset(json_file=train_json, root_dir=train_images, annotation_json=annotation_json)
EOS_TOKEN = 1
fig = plt.figure()

device = torch.device("cpu")
question_lang = readQuestions(train_json)

def prepare_image(image_cv2, do_normalize=True):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])
    # Resize
    img = cv2.resize(image_cv2, (224, 224))
    img = img[:, :, ::-1].copy()
    # Convert to tensor
    tensor_img = transforms.functional.to_tensor(img)

    # Possibly normalize
    if do_normalize:
        tensor_img = normalize(tensor_img)
    # Put image in a batch
    batch_tensor_img = torch.unsqueeze(tensor_img, 0)

    # Put the image in the gpu
    #if cuda_available:
    #    batch_tensor_img = batch_tensor_img.cuda()
    return batch_tensor_img

for i in range(len(vqa_dataset)): 
    sample = vqa_dataset[i]

    if i == 9:
        print(i, sample['image'].shape, sample['question'], sample["answer"])
        normalize_question = normalizeString(sample['question'])
        print(normalize_question)
        indexes = indexesFromSentence(question_lang, normalize_question)
        indexes.append(EOS_TOKEN)
        tensor = torch.tensor(indexes, dtype=torch.long, device=device).view(-1, 1)
        img = sample["image"]
        tensor_img = prepare_image(img)
        print(tensor)
        print(tensor_img)
        #plt.imshow(sample["image"])
        #plt.show()
        break
