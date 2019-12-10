import torchvision.transforms as transforms
import random
from tqdm import tqdm
from tensorboardX import SummaryWriter
import re
import os
import json
import torch
import pandas as pd
import numpy as np
import copy
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
        amr = question["amr"]

        if self.transform:
            image = self.transform(image)
            image = image[:-1,:,:] 
            image = image.to(DEVICE)


        sample = {'image': image, 'question': question_txt, 
                  "answer": answer_txt, "amr": amr}

        return sample

class Lang:
    def __init__(self, name):
        self.SOS_TOKEN = 0
        self.EOS_TOKEN = 1
        self.PAD_TOKEN = 2
        self.UNK_TOKEN = 3
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
    s = re.sub(r"([.!?()])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?0-9:\-]+", r" ", s)
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
    amr_lang = Lang("amr")
    answer_lang_temp = Lang("answers")
    #for idx in range(len(questions)):
    for idx in range(len(data_df)):
        sample = data_df.iloc[idx]
        answer = sample["multiple_choice_answer"]
        question = sample["question"]
        amr = sample["amr"]
        question_lang.addSentence(normalizeString(question))
        amr_lang.addSentence(normalizeString(amr))
        answer_lang_temp.addSentence(normalizeAnswerString(answer))

    most_common_answers = sorted(answer_lang_temp.word2count.items(), key = lambda x: x[1])
    most_common_answers = list(reversed(most_common_answers))
    answer_lang = Lang("answers")
    for answer in most_common_answers[:996]:
        answer_lang.addWord(answer[0])

    return question_lang, answer_lang, amr_lang

def batch_answers(answers, answer_lang):
    tensors = []
    normalized_answers = [normalizeAnswerString(a) for a in answers]
    answer_idxs = [indexesFromSentence(answer_lang, a) for a in normalized_answers]
    tensors = [torch.tensor(i, dtype=torch.long, device=DEVICE) for i in answer_idxs]
    tensors = torch.cat(tensors)
    return tensors

def batch_text(texts, text_lang, EOS_TOKEN=False, SOS_TOKEN=False):
    tensors = []
    normalized_texts = [normalizeString(q) for q in texts]
    text_idxs = [indexesFromSentence(text_lang, q) for q in normalized_texts]
    lengths = [len(q) + EOS_TOKEN + SOS_TOKEN for q in text_idxs] 
    max_len = max(lengths)
    for indexes in text_idxs:
        if SOS_TOKEN:
            indexes.insert(0, text_lang.SOS_TOKEN)
        if EOS_TOKEN:
            indexes.append(text_lang.EOS_TOKEN)
        indexes = indexes + [text_lang.PAD_TOKEN]*(max_len-len(indexes))
        tensors.append(torch.tensor(indexes, dtype=torch.long, device=DEVICE))
    tensors = torch.stack(tensors)
    return tensors, lengths

class Decoder(nn.Module):
    """A conditional RNN decoder with attention."""
    
    def __init__(self, emb_size, hidden_size, softmax, trg_embed,
                 attention=None, num_layers=1, dropout=0.5,
                 bridge=True):
        super(Decoder, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.attention = attention
        self.dropout = dropout
        self.softmax = softmax
        self.trg_embed = trg_embed

        #the LSTM takes                
        self.rnn = nn.LSTM(emb_size + 2*hidden_size, hidden_size, num_layers,
                          batch_first=True, dropout=dropout)
                 
        # to initialize from the final encoder state
        self.dropout_layer = nn.Dropout(p=dropout)

        #input is prev embedding (emb_size), output (hidden_size), and context (num_directions*hidden_size)
        #output is a vector of size (hidden_size). This vector is fed through a softmax layer to get a 
        #distribution over vocab
        self.pre_output_layer = nn.Linear(hidden_size + 2*hidden_size + emb_size,
                                          hidden_size, bias=False)
        
    def forward_step(self, prev_embed, encoder_hidden, src_lengths, proj_key, hidden):
        """Perform a single decoder step (1 word)
           prev_embed (batch_size x seq_len x emb_size) is the target previous word
           encoder_hidden is atuple of elements with size (batch_size x seq_len x num_directions*hidden_size) is the output of the encoder
           hidden (batch_size x 1 x hidden_size) is the forward hidden state of the previous time step
        """
        # compute context vector using attention mechanism
        #we only want the hidden, not the cell state of the lstm CZW, hence the hidden[0]
        query = hidden[0][-1].unsqueeze(1)  # [#layers, B, D] -> [B, 1, D]
        context, attn_probs = self.attention(
            query=query, proj_key=proj_key,
            value=encoder_hidden, src_lengths=src_lengths)

        # update rnn hidden state
        # context is batch x 1 x num_directions*hidden_size
        # the lstm takes the previous target embedding and the attention context as input
        rnn_input = torch.cat([prev_embed, context], dim=2)
        output, hidden = self.rnn(rnn_input, hidden)
        
        pre_output = torch.cat([prev_embed, output, context], dim=2)
        pre_output = self.dropout_layer(pre_output)
        pre_output = self.pre_output_layer(pre_output)

        # at this stage, pre_output batch x seq_len x hidden_size
        # output is batch x seq_len x hidden_size 
        # hidden is a tuple of final hidden_cell, hidden_state
        # pre_output is actually used to compute the prediction
        return output, hidden, pre_output
    
    def forward(self, trg, encoder_hidden, 
                src_lengths, hidden, teacher_forcing_ratio=None,
                max_len=None):
        """Unroll the decoder one step at a time.
           trg is (batch_len x trg_max_seq_len)
           encoder_hidden is (batch_len x src_max_seq_len x num_directions*hidden_layer_size)
           encoder_final is a tuple of final hidden and final cell state. 
             each state is (num_layers x batch x num_directions*hidden_layer_size)
           src_lengths is (batch_len x src_max_seq_len)

           returns a list of distributions - one per timestep and the final hidden state
        """
                                         
        # the maximum number of steps to unroll the RNN
        if max_len is None:
            max_len = trg.size(1)

        # pre-compute projected encoder hidden states
        # (the "keys" for the attention mechanism)
        # this is only done for efficiency
        proj_key = self.attention.key_layer(encoder_hidden)
        out = torch.zeros((trg.shape[0], self.trg_embed.num_embeddings, max_len), device=trg.device)
        # unroll the decoder RNN for max_len steps
        teacher_forcing = random.random() < teacher_forcing_ratio
        for i in range(max_len):
            if i == 0:
                prev_y = trg[:,0].unsqueeze(1)
            
            #print(prev_y)
            prev_embed = self.trg_embed(prev_y)
            output, hidden, pre_output = self.forward_step(prev_embed,
                                                           encoder_hidden, 
                                                           src_lengths,
                                                           proj_key,
                                                           hidden)

            prob = self.softmax(pre_output).squeeze(1)
            #print(prob.shape)
            out[:,:,i] = prob
            if not teacher_forcing:
                words = prob.topk(1, dim=1)[1]
                prev_y = words.detach()
            elif i < max_len-1:
                prev_y = trg[:, i+1].unsqueeze(1)
        return out, hidden


class BahdanauAttention(nn.Module):
    """Implements Bahdanau (MLP) ttention"""
    
    def __init__(self, hidden_size, key_size=None, query_size=None):
        super(BahdanauAttention, self).__init__()
        
        # We assume a bi-directional encoder so key_size is 2*hidden_size
        key_size = 2 * hidden_size if key_size is None else key_size
        query_size = hidden_size if query_size is None else query_size

        self.key_layer = nn.Linear(key_size, hidden_size, bias=False)
        self.query_layer = nn.Linear(query_size, hidden_size, bias=False)
        self.energy_layer = nn.Linear(hidden_size, 1, bias=False)
        
        # to store attention scores
        self.alphas = None
        
    def forward(self, query=None, proj_key=None, value=None, src_lengths=None):
        # We first project the query (the decoder state).
        # The projected keys (the encoder states) were already pre-computated.
        query = self.query_layer(query)
        
        # Calculate scores.
        scores = self.energy_layer(torch.tanh(query + proj_key))
        scores = scores.squeeze(2).unsqueeze(1)
        
        # Mask out invalid positions.
        # The mask marks valid positions so we invert it using `mask & 0`.
        mask = []
        max_len = scores.size(2)
        for length in src_lengths:
            mask.append([[1]*length + [0]*(max_len - length)])

        tensor_mask = torch.Tensor(mask).ne(1)
        try:
            tensor_mask = tensor_mask.to(DEVICE)
        except RuntimeError as e:
            print(e)
            print(mask)
            print(tensor_mask.shape)
            exit()
     
        scores.data.masked_fill_(tensor_mask, -float('inf'))
        
        # Turn scores to probabilities.
        alphas = F.softmax(scores, dim=-1)
        self.alphas = alphas        
        
        # The context vector is the weighted sum of the values.
        context = torch.bmm(alphas, value)
        # context shape: [B, 1, 2D], alphas shape: [B, 1, M]
        return context, alphas

class VQA_Model(nn.Module):
    def __init__(self, input_size, emb_size, hidden_size, num_layers, 
                 num_answers, resnet, decoder, dropout): 
        super(VQA_Model, self).__init__()
        self.num_layers = num_layers
        self.embedding = nn.Embedding(input_size, emb_size)
        self.rnn = nn.LSTM(emb_size, hidden_size, num_layers, 
                          batch_first=True, bidirectional=True, dropout=dropout)
        self.proj_q = nn.Linear(hidden_size, hidden_size, bias=False)

        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True)
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) 

        model_cut  = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = model_cut

        self.proj_im_1 = nn.Linear(2048, hidden_size, bias=False)
        #self.proj_im_2 = nn.Linear(2*hidden_size, 2*hidden_size)

        self.q_im_embed = nn.Linear(hidden_size, hidden_size, bias=False)
        self.fc1 = nn.Linear(hidden_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, num_answers)
        self.decoder = decoder 

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return (torch.tanh(self.bridge_hidden(encoder_final[0])),
                torch.tanh(self.bridge_cell(encoder_final[1])))

    def embed_question(self, question, lengths):
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
        encoder_final = (final_hidden, final_cell)
        return output, encoder_final

    def embed_image(self, image):
        im_embed = self.resnet(image)
        im_embed = im_embed.squeeze(-1)
        im_embed = im_embed.squeeze(-1)#shape is [batch, 2048, 1, 1] before squeezing
        im_embed = torch.relu(self.proj_im_1(im_embed))
        #im_embed = self.proj_im_2(im_embed)
        return im_embed

    def forward(self, question, lengths, image, amr, teacher_forcing_ratio=None):
        encoder_hidden, encoder_final = self.embed_question(question, lengths)
        bridge_hidden = self.init_hidden(encoder_final)
        q_embed = torch.relu(self.proj_q(bridge_hidden[0][0]))

        im_embed = self.embed_image(image)

        q_im_embed = torch.mul(q_embed, im_embed)
        bridge_hidden = (q_im_embed.unsqueeze(0), bridge_hidden[0])

        q_im_embed = torch.relu(self.fc1(q_im_embed))
        #q_im_embed = torch.tanh(self.fc2(q_im_embed))
        #q_im_embed = torch.tanh(self.fc3(q_im_embed))
        pred = F.log_softmax(q_im_embed, dim=-1)
        amr_probs, _ = self.decoder(amr, encoder_hidden,
                                    lengths, bridge_hidden,
                                    teacher_forcing_ratio=teacher_forcing_ratio)

        amr_tokens = torch.max(amr_probs.transpose(2,1), dim=2)[1]
        amr_tokens = amr_tokens.cpu().detach()
            
        return pred, amr_probs, amr_tokens

def run_epoch(model, vqa_dataloader, question_lang, answer_lang, amr_lang, criterion, optim, teacher_forcing_ratio=None, alpha=1.0):
    total_loss = 0
    for idx, batch in enumerate(vqa_dataloader):
        img = batch["image"]
        question_tensor, lengths = batch_text(batch['question'], question_lang, EOS_TOKEN=True, SOS_TOKEN=False)
        amr_tensor, amr_lengths = batch_text(batch['amr'], amr_lang, EOS_TOKEN=False, SOS_TOKEN=True)
        amr_tensor_trg, amr_lengths_trg = batch_text(batch['amr'], amr_lang, EOS_TOKEN=True, SOS_TOKEN=False)
        out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, 
                                                   amr_tensor, teacher_forcing_ratio=teacher_forcing_ratio)
        answer_tensors = batch_answers(batch["answer"], answer_lang)
        assert max(answer_tensors) < 1000
        loss = criterion(out, answer_tensors)
        loss += alpha*criterion(amr_probs, amr_tensor_trg)
        total_loss += loss.item()

        loss.backward()          
        optim.step()
        optim.zero_grad()
    return total_loss

def eval_accuracy(model, vqa_dataloader, question_lang, answer_lang, amr_lang, out_file=None):
    count, amr_count = 0.0, 0.0
    denom = 0
    stripEOS = lambda x: re.sub("{}.*".format("EOS"), "", x)
    if out_file:
        f = open(out_file, "w")
    for idx, batch in enumerate(vqa_dataloader):
        question_tensor, lengths = batch_text(batch['question'], question_lang, EOS_TOKEN=True, SOS_TOKEN=False)
        img = batch["image"]
        question_tensor, lengths = batch_text(batch['question'], question_lang, EOS_TOKEN=True, SOS_TOKEN=False)
        amr_tensor, amr_lengths = batch_text(batch['amr'], amr_lang, EOS_TOKEN=False, SOS_TOKEN=True)

        with torch.no_grad():
            out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, amr_tensor, teacher_forcing_ratio=0.0)
        max_tokens = torch.max(out,1)[1]
        pred_answers = [answer_lang.getWord(x.item()) for x in max_tokens]
        normalized_answers = [normalizeAnswerString(x) for x in batch["answer"]]
        count += sum([x==y for x,y in zip(pred_answers, normalized_answers)])
        #print([x==y for x,y in zip(pred_answers, normalized_answers)])
        #print(list(zip(pred_answers, normalized_answers)))
        trg_amrs = batch["amr"]
        pred_amrs = [' ' .join(indicesToWords(amr_lang, x)) for x in amr_tokens]
        pred_amrs = [stripEOS(x) for x in pred_amrs]
        amr_count += sum([x==y for x,y in zip(pred_amrs, trg_amrs)])
        denom += len(pred_amrs)
        #print(list(zip(pred_amrs, trg_amrs)))

        if out_file:
            for i in range(len(max_tokens)):
                f.write(batch["question"][i] + "\n")
                f.write(normalized_answers[i] + "\n")
                f.write(pred_answers[i] + "\n")
                f.write(trg_amrs[i] + "\n")
                f.write(pred_amrs[i] + "\n\n")
    if out_file:
        f.close()
    return count/denom, amr_count/denom

def indicesToWords(lang, sentence):
    return [lang.getWord(int(x)) for x in sentence]

def print_examples(model, vqa_dataloader, question_lang, answer_lang, amr_lang, num_examples=3):
    stripEOS = lambda x: re.sub("{}.*".format("EOS"), "", x)
    for idx, batch in enumerate(vqa_dataloader):
        question_tensor, lengths = batch_text(batch['question'], question_lang, EOS_TOKEN=True, SOS_TOKEN=False)
        amr_tensor, amr_lengths = batch_text(batch['amr'], amr_lang, EOS_TOKEN=False, SOS_TOKEN=True)
        img = batch["image"]
        with torch.no_grad():
            out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, amr_tensor, teacher_forcing_ratio=0.0)
        max_prob, max_tokens = torch.max(out,1)
        #print(torch.exp(max_prob))
        #print(torch.exp(out[0]))

        for count in range(len(batch['question'])):
            print(f'\nExample {count}')
            print("question: ", batch["question"][count])
            print("answer: ", batch["answer"][count])
            max_token = max_tokens[count].item()
            print("pred answer: ", answer_lang.getWord(max_token))
            print("amr: ", batch["amr"][count])
            pred_amr = ' '.join(indicesToWords(amr_lang, amr_tokens[count]))
            pred_amr = stripEOS(pred_amr)
            print("pred amr: ", pred_amr)
        
            if count >= num_examples:
                return

def train(model, vqa_dataloader, val_dataloader, question_lang, answer_lang, amr_lang, num_epochs, optim, alpha):
    criterion = nn.NLLLoss(reduction="mean", ignore_index=amr_lang.PAD_TOKEN)
    max_acc = 0
    best_model_wts = None
    for epoch in range(num_epochs):
        print(f'\n***Epoch {epoch}***')
        model.train()
        loss = run_epoch(model, vqa_dataloader, question_lang, answer_lang, amr_lang, criterion, optim, teacher_forcing_ratio=1.0, alpha=alpha)
        writer.add_scalar("loss", loss, epoch)
        print("loss", loss)
        model.eval()
        with torch.no_grad():
            print_examples(model, val_dataloader, question_lang, answer_lang, amr_lang)
            #val_accuracy, val_accuracy_amr = eval_accuracy(model, val_dataloader, question_lang, answer_lang, amr_lang)
            #print("val_accuracy", val_accuracy)
            #print("val_accuracy_amr", val_accuracy_amr)
            #writer.add_scalar("val_accuracy", val_accuracy, epoch)
            #writer.add_scalar("val_accuracy_amr", val_accuracy_amr, epoch)
            #if val_accuracy >= max_acc:
            #    max_acc = val_accuracy
            #TODO indent the below line
            best_model_wts = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_model_wts)
    return model

#TODO indent the below line
# To keep things easy we also keep the `Softmax` class the same. 
# It simply projects the pre-output layer ($x$ in the `forward` function below) to obtain the output layer, so that the final dimension is the target vocabulary size.
class Softmax(nn.Module):
    """Define standard linear + softmax generation step."""
    def __init__(self, hidden_size, vocab_size):
        super(Softmax, self).__init__()
        self.proj = nn.Linear(hidden_size, vocab_size, bias=False)

    def forward(self, x):
        """
            In most cases, x is (batch_len x seq_len x hidden_size)
        """
        return F.log_softmax(self.proj(x), dim=-1)

def make_model(question_lang, amr_lang, num_answers):
    resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=True) 

    num_layers = 1
    emb_size = 200
    dropout = 0.0
    hidden_size = cfg["hidden_size"]
    question_vocab_size = question_lang.n_words
    amr_vocab_size = amr_lang.n_words
    attention = BahdanauAttention(hidden_size)
    decoder = Decoder(emb_size, hidden_size, Softmax(hidden_size, amr_vocab_size), 
                      nn.Embedding(amr_vocab_size, emb_size),
                      attention, num_layers=num_layers, dropout=dropout)
    model = VQA_Model(question_lang.n_words, emb_size, hidden_size, num_layers,
                      num_answers, resnet, decoder, dropout=dropout)
    return model.to(DEVICE)

def make_df(json_file, annotation_json, amr_file):
    amr = []
    with open(amr_file, 'r') as f:
        lines = f.readlines()
        for line in lines:
            amr.append(line.rstrip('\n'))


    data = json.load(open(json_file))
    questions = pd.DataFrame(data["questions"])
    anno_data = json.load(open(annotation_json))
    annotations = pd.DataFrame(anno_data["annotations"])
    data_df = questions.join(annotations.set_index('question_id'), on='question_id', rsuffix="_anno")
    data_df["amr"] = amr
    return data_df

def make_splits():
    train_json = 'Questions_Train_abstract_v002/MultipleChoice_abstract_v002_train2015_questions.json'
    val_json = 'Questions_Val_abstract_v002/MultipleChoice_abstract_v002_val2015_questions.json'
    train_df = make_df(train_json, 'abstract_v002_train2015_annotations.json', 'train_amr.txt')
    val_df = make_df(val_json, 'abstract_v002_val2015_annotations.json', 'val_amr.txt')
    
    train_df = train_df.iloc[:cfg["num_train_examples"]]
    n_val = int(len(val_df)*cfg["val_test_split"])
    val_df_temp = val_df.iloc[:n_val]
    test_df = val_df.iloc[n_val:]
    return train_df, val_df_temp, test_df

def main():
    transform = transforms.Compose([transforms.Resize(224),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(),
                                    #transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                         #std=[0.229, 0.224, 0.225])
                                   ])
    train_df, val_df, test_df = make_splits()
    train_images = 'scene_img_abstract_v002_train2015'
    val_images = 'scene_img_abstract_v002_val2015'
    train_dataset = VQADataset(train_df, root_dir=train_images, split="train",
                               transform=transform)
    val_dataset = VQADataset(val_df, root_dir=val_images, split="val",
                             transform=transform)
    test_dataset = VQADataset(test_df, root_dir=val_images, split="val",
                             transform=transform)
    question_lang, answer_lang, amr_lang = readQAText(train_df)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=cfg["batch_size"])
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=cfg["val_batch_size"])
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=cfg["val_batch_size"])
    model = make_model(question_lang, amr_lang, 1000)
    optim = torch.optim.Adam(model.parameters(), lr=cfg["lr"])
    model = train(model, train_loader, val_loader, question_lang, answer_lang,
          amr_lang, cfg["num_epochs"], optim, cfg["alpha"])
    model.eval()
    val_accuracy, val_accuracy_amr = eval_accuracy(model, val_loader, question_lang, answer_lang, amr_lang, f'{OUT_DIR}/{EXP_NAME}/val_pred.txt')
    train_accuracy, train_accuracy_amr = eval_accuracy(model, train_loader, question_lang, answer_lang, amr_lang, f'{OUT_DIR}/{EXP_NAME}/train_pred.txt')
    test_accuracy, test_accuracy_amr = eval_accuracy(model, test_loader, question_lang, answer_lang, amr_lang, f'{OUT_DIR}/{EXP_NAME}/test_pred.txt')
    print(f'val {val_accuracy}')
    print(f'train {train_accuracy}')
    print(f'test {test_accuracy}')
    print(f'val amr {val_accuracy_amr}')
    print(f'train amr {train_accuracy_amr}')
    print(f'test amr {test_accuracy_amr}')

    with open(f'{OUT_DIR}/{EXP_NAME}/results.txt', "w") as f:
        f.write(f'val {val_accuracy}\n')
        f.write(f'train {train_accuracy}\n')
        f.write(f'test {test_accuracy}\n')
        f.write(f'val amr {val_accuracy_amr}\n')
        f.write(f'train amr {train_accuracy_amr}\n')
        f.write(f'test amr {test_accuracy_amr}\n')


if __name__ == '__main__':
    main()
