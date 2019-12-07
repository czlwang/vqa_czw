import torchvision.transforms as transforms
from tqdm import tqdm
from tensorboardX import SummaryWriter
import re
import os
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
import yaml
import pathlib

with open("config.yml", 'r') as ymlfile:
    cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    print(cfg)

OUT_DIR = cfg["out_dir"]
EXP_NAME = cfg["exp_name"]
pathlib.Path(f'{OUT_DIR}/{EXP_NAME}').mkdir(parents=True, exist_ok=True)

writer = SummaryWriter(f'runs/{OUT_DIR}/{EXP_NAME}')

with open(f'{OUT_DIR}/{EXP_NAME}/config.txt', "w") as f:
    for k, v in cfg.items():
        f.write(f'\n{k} : {v}')

USE_CUDA = torch.cuda.is_available()
ordinal = cfg["cuda"]
DEVICE = torch.device(f'cuda:{ordinal}') if USE_CUDA else torch.device('cpu')

class VQADataset(Dataset):
    """vqa dataset."""

    def __init__(self, data_df, root_dir, split="train", transform=None):
        """
        Args:
            json_file (string): Path to the json file with annotations.
            root_dir (string): Directory with all the images.
            annotation_json: contains the human answers
        """
        self.root_dir = root_dir
        self.transform = transform
        self.data_df = data_df
        self.split = split

    def __len__(self):
        return len(self.data_df)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        question = self.data_df.iloc[idx]
        image_id = question["image_id"]
        image_id = f'abstract_v002_{self.split}2015_{image_id:012}.png'
        question_txt = question["question"]
        question_id = question["question_id"]
        answer_txt = question["multiple_choice_answer"]
        img_name = os.path.join(self.root_dir, image_id)
        image = Image.open(img_name)

        if self.transform:
            image = self.transform(image)
            image = image[:-1,:,:] 
            image = image.to(DEVICE)

        sample = {'image': image, 'question': question_txt, "answer": answer_txt}

        return sample

class Lang:
    def __init__(self, name):
        self.SOS_TOKEN = 1
        self.EOS_TOKEN = 2
        self.PAD_TOKEN = 3
        self.UNK_TOKEN = 4
        self.name = name
        self.word2count = {}
        self.index2word = {self.SOS_TOKEN: "SOS", 
                           self.EOS_TOKEN: "EOS", 
                           self.PAD_TOKEN:"PAD", 
                           self.UNK_TOKEN:"UNK"}
        self.word2index = {v:k for k,v in self.index2word.items()}
        self.n_words = 4

    def getWord(self, idx):
        return self.index2word.get(idx, "UNK")

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
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    return s

def normalizeAnswerString(s):
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?0-9]+", r" ", s)
    s = re.sub(r"[\s]+", r"_", s)
    return s

def indexesFromSentence(lang, sentence):
    return [lang.word2index.get(word, lang.UNK_TOKEN) for word in sentence.split(' ')]

def readQAText(data_df):
    question_lang = Lang("questions")
    answer_lang_temp = Lang("answers")
    #for idx in range(len(questions)):
    for idx in range(len(data_df)):
        sample = data_df.iloc[idx]
        answer = sample["multiple_choice_answer"]
        question = sample["question"]
        question_lang.addSentence(normalizeString(question))
        answer_lang_temp.addSentence(normalizeAnswerString(answer))

    most_common_answers = sorted(answer_lang_temp.word2count.items(), key = lambda x: x[1])
    most_common_answers = list(reversed(most_common_answers))
    answer_lang = Lang("answers")
    for answer in most_common_answers[:996]:
        answer_lang.addWord(answer[0])

    return question_lang, answer_lang

def batch_answers(answers, answer_lang):
    tensors = []
    normalized_answers = [normalizeAnswerString(a) for a in answers]
    answer_idxs = [indexesFromSentence(answer_lang, a) for a in normalized_answers]
    tensors = [torch.tensor(i, dtype=torch.long, device=DEVICE) for i in answer_idxs]
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
        tensors.append(torch.tensor(indexes, dtype=torch.long, device=DEVICE))
    tensors = torch.stack(tensors)
    return tensors, lengths

class VQA_Model(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, 
                 num_answers, resnet, dropout): 
        super(VQA_Model, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.proj_q = nn.Linear(2*hidden_size, 2*hidden_size, bias=False)

        model_cut  = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = model_cut

        self.proj_im_1 = nn.Linear(2048, 2*hidden_size, bias=False)
        #self.proj_im_2 = nn.Linear(2*hidden_size, 2*hidden_size)

        self.q_im_embed = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.fc1 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size,2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, num_answers)

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
        bwd_final_hidden = final[0][1:final[0].size(0):2]
        final_hidden = torch.cat([fwd_final_hidden, bwd_final_hidden], dim=2)  # [num_layers, batch, 2*dim]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, 2*dim]
        top_final_cell = final_cell[1]
        q_embed = self.proj_q(top_final_cell)
        im_embed = self.resnet(image)
        im_embed = im_embed.squeeze(-1)
        im_embed = im_embed.squeeze(-1)#shape is [batch, 2048, 1, 1] before squeezing
        im_embed = self.proj_im_1(im_embed)
        #im_embed = self.proj_im_2(im_embed)

        #concat = torch.cat((q_embed, im_embed), dim=1)
        q_im_embed = torch.mul(q_embed, im_embed)
        #q_im_embed = q_embed
        q_im_embed = self.fc1(q_im_embed)
        q_im_embed = self.fc2(q_im_embed)
        q_im_embed = self.fc3(q_im_embed)
        pred = F.log_softmax(q_im_embed, dim=-1)
        return pred

def run_epoch(model, vqa_dataloader, question_lang, answer_lang, criterion, optim):
    total_loss = 0
    for idx, batch in enumerate(vqa_dataloader):
        img = batch["image"]
        question_tensor, lengths = batch_questions(batch['question'], question_lang)
        out = model.forward(question_tensor, lengths, img)
        answer_tensors = batch_answers(batch["answer"], answer_lang)
        assert max(answer_tensors) < 1000
        loss = criterion(out, answer_tensors)
        total_loss += loss.item()

        loss.backward()          
        optim.step()
        optim.zero_grad()
    return total_loss

def eval_accuracy(model, vqa_dataloader, question_lang, answer_lang, save_file=None):
    """
    assume batches are size 1
    """
    count = 0.0
    with open(save_file, "w") as f:
        for idx, batch in enumerate(vqa_dataloader):
            question_tensor, lengths = batch_questions(batch['question'], question_lang)
            img = batch["image"]
            question_tensor, lengths = batch_questions(batch['question'], question_lang)
            with torch.no_grad():
                out = model.forward(question_tensor, lengths, img)
            max_token = torch.max(out,1)[1].item()
            pred_answer = answer_lang.getWord(max_token)
            normalized_answer = normalizeAnswerString(batch["answer"][0])
            f.write(batch["question"][0] + "\n")
            f.write(normalized_answer + "\n")
            f.write(pred_answer + "\n\n")
            if normalized_answer == pred_answer:
                count += 1 
    return count/len(vqa_dataloader) 
 
def print_examples(model, vqa_dataloader, question_lang, answer_lang, num_examples=3):
    """
    assume batches are size 1
    """
    for idx, batch in enumerate(vqa_dataloader):
        question_tensor, lengths = batch_questions(batch['question'], question_lang)
        print(f'Example {idx}')
        print("question: ", batch["question"])
        print("answer: ", batch["answer"])
        img = batch["image"]
        question_tensor, lengths = batch_questions(batch['question'], question_lang)
        with torch.no_grad():
            out = model.forward(question_tensor, lengths, img)
        max_token = torch.max(out,1)[1].item()
        print("pred: ", answer_lang.getWord(max_token))
        
        if idx >= num_examples:
            break

def train(model, vqa_dataloader, val_dataloader, question_lang, answer_lang, num_epochs, optim):
    criterion = nn.NLLLoss(reduction="sum")
    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch}')
        model.train()
        loss = run_epoch(model, vqa_dataloader, question_lang, answer_lang, criterion, optim)
        writer.add_scalar("loss", loss, epoch)
        print("loss", loss)
        model.eval()
        with torch.no_grad():
            print_examples(model, val_dataloader, question_lang, answer_lang)
            val_accuracy = eval_accuracy(model, val_dataloader, question_lang, answer_lang, "/tmp/eval.txt")
            print("val_accuracy", val_accuracy)
            writer.add_scalar("val_accuracy", val_accuracy, epoch)

def make_model(question_lang, num_answers):
    resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=True) 

    num_layers = 2
    emb_size = 200
    dropout = 0.5
    hidden_size = cfg["hidden_size"]
    model = VQA_Model(question_lang.n_words, emb_size, hidden_size, num_layers,
                      num_answers, resnet, dropout=dropout)
    return model.to(DEVICE)

def make_df(json_file, annotation_json):
    data = json.load(open(json_file))
    questions = pd.DataFrame(data["questions"])
    anno_data = json.load(open(annotation_json))
    annotations = pd.DataFrame(anno_data["annotations"])
    data_df = questions.join(annotations.set_index('question_id'), on='question_id', rsuffix="_anno")
    return data_df

def main():
    train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
    val_json = 'Questions_Val_abstract_v002/MultipleChoice_abstract_v002_val2015_questions.json'
    train_images = 'scene_img_abstract_v002_train2015'
    val_images = 'scene_img_abstract_v002_val2015'
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
#                                                         std=[0.229, 0.224, 0.225])
                                   ])
    train_df = make_df(train_json, 'abstract_v002_train2015_annotations.json')
    val_df = make_df(val_json, 'abstract_v002_val2015_annotations.json')
    
    train_df = train_df.iloc[:cfg["num_train_examples"]]
    val_df = val_df.iloc[:cfg["num_val_examples"]]

    train_dataset = VQADataset(train_df, root_dir=train_images, split="train",
                               transform=transform)
    val_dataset = VQADataset(val_df, root_dir=val_images, split="val",
                             transform=transform)
    #TODO change val to actually val
    #val_dataset = VQADataset(train_df, root_dir=train_images, split="train",
    #                         transform=transform)

    question_lang, answer_lang = readQAText(train_df)
    vqa_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"])
    val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size=1)
    model = make_model(question_lang, 1000)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    train(model, vqa_dataloader, val_dataloader, question_lang, 
          answer_lang, cfg["num_epochs"], optim)
    model.eval()
    vqa_dataloader_1_batch = torch.utils.data.DataLoader(train_dataset, batch_size=1)
    val_accuracy = eval_accuracy(model, val_dataloader, question_lang, answer_lang, f'{OUT_DIR}/{EXP_NAME}/val_pred.txt')
    train_accuracy = eval_accuracy(model, vqa_dataloader_1_batch, question_lang, answer_lang, f'{OUT_DIR}/{EXP_NAME}/train_pred.txt')
    print(f'val {val_accuracy}')
    print(f'train {train_accuracy}')

if __name__ == '__main__':
    main()
