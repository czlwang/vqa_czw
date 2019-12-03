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
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torchvision.models as models
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms, utils
from PIL import Image

device = torch.device("cpu")

class VQADataset(Dataset):
    """vqa dataset."""

    def __init__(self, data_df, root_dir, transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            annotation_json: contains the human answers
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_df = data_df

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        question = self.data_df.iloc[idx]
        image_id = question["image_id"]
        image_id = f'abstract_v002_train2015_{image_id:012}.png'
        question_txt = question["question"]
        question_id = question["question_id"]
        answer_txt = question["multiple_choice_answer"]
        img_name = os.path.join(self.root_dir, image_id)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
            image = image[:-1,:,:] 

        sample = {'image': image, 'question': question_txt, "answer": answer_txt}

        return sample

class Lang:
    def __init__(self, name):
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2
        self.PAD_TOKEN = 3
        self.UNK_TOKEN = 4
        self.name = name
        self.word2index = {}
        self.word2count = {}
        self.index2word = {self.SOS_TOKEN: "SOS", 
                           self.EOS_TOKEN: "EOS", 
                           self.PAD_TOKEN:"PAD", 
                           self.UNK_TOKEN:"UNK"}
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
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    return s

def normalizeAnswerString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?]+", r" ", s)
    s = re.sub(r"[\s]+", r"_", s)
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, 2) for word in sentence.split(' ')]

def readQAText(data_df):
    question_lang = Lang("questions")
    answer_lang = Lang("answers")
    #for idx in range(len(questions)):
    for idx in range(10):
        sample = data_df.iloc[idx]
        answer = sample["multiple_choice_answer"]
        question = sample["question"]
        question_lang.addSentence(normalizeString(question))
        answer_lang.addSentence(normalizeAnswerString(answer))
    return question_lang, answer_lang

def batch_answers(answers, answer_lang):
    tensors = []
    normalized_answers = [normalizeString(a) for a in answers]
    answer_idxs = [indexesFromSentence(answer_lang, a) for a in normalized_answers]
    tensors = [torch.tensor(i, dtype=torch.long, device=device) for i in answer_idxs]
    tensors = torch.cat(tensors)
    return tensors

def batch_questions(questions, question_lang):
    tensors = []
    normalized_questions = [normalizeString(q) for q in questions]
    question_idxs = [indexesFromSentence(question_lang, q) for q in normalized_questions]
    lengths = [len(q) + 1 for q in question_idxs] 
    max_len = max(lengths)
    for indexes in question_idxs:
        indexes.append(question_lang.EOS_TOKEN)
        indexes = indexes + [question_lang.PAD_TOKEN]*(max_len-len(indexes))
        tensors.append(torch.tensor(indexes, dtype=torch.long, device=device))
    tensors = torch.stack(tensors)
    return tensors, lengths

class VQA_Model(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, num_answers, dropout): 
        super(VQA_Model, self).__init__()
        self.num_layers = num_layers
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.proj_q = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.proj_im = nn.Linear(2048, hidden_size, bias=False)
        self.q_im_embed = nn.Linear(2*hidden_size, num_answers, bias=False)
        self.embedding = nn.Embedding(input_size, emb_size)
        resnet = models.resnet50(num_classes=365)
        model_cut  = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = model_cut

    def forward(self, question, lengths, image):
        """
        Applies a bidirectional LSTM
        The input mini-batch x needs to be sorted by length.
        x should have dimensions [batch, time, dim].
        """
        embedded = self.embedding(question)
        packed = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        output, final = self.rnn(packed)
        output, _ = pad_packed_sequence(output, batch_first=True)
        
        # output is of size (batch_length, seq_length, num_directions*hidden_size)
        # final[0] is of size (num_layers*num_directions, batch_length, hidden_size)
        # final[0][0] is fwd for first layer. final[0][2] is forward for second layer.
        # # we need to manually concatenate the final states for both directions
        # final is a tuple of (hidden, cell)
        fwd_final_hidden = final[0][0:final[0].size(0):2]
        #print("final_fwd size")
        #print(fwd_final_hidden.size())
        bwd_final_hidden = final[0][1:final[0].size(0):2]
        final_hidden = torch.cat([fwd_final_hidden, bwd_final_hidden], dim=2)  # [num_layers, batch, 2*dim]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, 2*dim]
        top_final_cell = final_cell[1]
        q_embed = self.proj_q(top_final_cell)
        im_embed = self.resnet(image).squeeze()
        im_embed = self.proj_im(im_embed)

        concat = torch.cat((q_embed, im_embed), dim=1)
        q_im_embed = self.q_im_embed(concat)
        pred = F.log_softmax(q_im_embed, dim=-1)
        return pred

def run_epoch(model, vqa_dataloader, question_lang, answer_lang, criterion, optim):
    loss = 0
    for batch in vqa_dataloader:
        img = batch["image"]
        question_tensor, lengths = batch_questions(batch['question'], question_lang)
        out = model.forward(question_tensor, lengths, img)
        answer_tensors = batch_answers(batch["answer"], answer_lang)
        loss = criterion(out, answer_tensors)
        loss += loss
        break
    loss.backward()          
    optim.step()
    optim.zero_grad()

def train(model, vqa_dataloader, question_lang, answer_lang, num_epochs, optim):
    criterion = nn.NLLLoss(reduction="sum")
    for epoch in range(num_epochs):
        model.train()
        run_epoch(model, vqa_dataloader, question_lang, answer_lang, criterion, optim)
        break

def make_model(question_lang, num_answers):
    hidden_size = 200
    num_layers = 2
    emb_size = 200
    dropout = 0.5
    model = VQA_Model(question_lang.n_words, emb_size, hidden_size, num_layers, num_answers, dropout=dropout)
    return model

def main():
    train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
    train_images = 'scene_img_abstract_v002_train2015'
    annotation_json = 'abstract_v002_train2015_annotations.json'
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225])
                                   ])
    data = json.load(open(train_json))
    questions = pd.DataFrame(data["questions"])
    anno_data = json.load(open(annotation_json))
    annotations = pd.DataFrame(anno_data["annotations"])
    data_df = questions.join(annotations.set_index('question_id'), on='question_id', rsuffix="_anno")

    vqa_dataset = VQADataset(data_df,
                             root_dir=train_images, 
                             transform=transform)

    question_lang, answer_lang = readQAText(data_df)
    vqa_dataloader = torch.utils.data.DataLoader(vqa_dataset, batch_size=4)
    model = make_model(question_lang, 1000)
    optim = torch.optim.Adam(model.parameters(), lr=10e-3)
    num_epochs = 10
    train(model, vqa_dataloader, question_lang, answer_lang, num_epochs, optim)

if __name__ == '__main__':
    main()
