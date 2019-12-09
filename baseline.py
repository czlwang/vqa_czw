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

def batch_text(texts, text_lang):
    tensors = []
    normalized_texts = [normalizeString(q) for q in texts]
    text_idxs = [indexesFromSentence(text_lang, q) for q in normalized_texts]
    lengths = [len(q) + 1 for q in text_idxs] 
    max_len = max(lengths)
    for indexes in text_idxs:
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
        self.bridge_hidden = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None
        self.bridge_cell = nn.Linear(2*hidden_size, hidden_size, bias=True) if bridge else None

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
    
    def forward(self, trg, encoder_hidden, encoder_final, 
                src_lengths, teacher_forcing_ratio=1.0,
                hidden=None, max_len=None):
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

        # initialize decoder hidden state
        if hidden is None:
            hidden = self.init_hidden(encoder_final)
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
            
            prev_embed = self.trg_embed(prev_y)
            output, hidden, pre_output = self.forward_step(prev_embed,
                                                           encoder_hidden, 
                                                           src_lengths,
                                                           proj_key,
                                                           hidden)

            prob = self.softmax(pre_output).squeeze(1)
            out[:,:,i] = prob
            if not teacher_forcing:
                words = prob.topk(1, dim=1)[1]
                prev_y = words.detach()
            elif i < max_len-1:
                prev_y = trg[:, i+1].unsqueeze(1)
        return out, hidden

    def init_hidden(self, encoder_final):
        """Returns the initial decoder state,
        conditioned on the final encoder state."""

        if encoder_final is None:
            return None  # start with zeros

        return (torch.tanh(self.bridge_hidden(encoder_final[0])),
                torch.tanh(self.bridge_cell(encoder_final[1])))

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
        tensor_mask = tensor_mask.to(DEVICE)
     
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
        self.proj_q = nn.Linear(2*hidden_size, 2*hidden_size, bias=False)

        model_cut  = nn.Sequential(*list(resnet.children())[:-1])
        self.resnet = model_cut

        self.proj_im_1 = nn.Linear(2048, 2*hidden_size, bias=False)
        #self.proj_im_2 = nn.Linear(2*hidden_size, 2*hidden_size)

        self.q_im_embed = nn.Linear(2*hidden_size, hidden_size, bias=False)
        self.fc1 = nn.Linear(2*hidden_size, 2*hidden_size)
        self.fc2 = nn.Linear(2*hidden_size,2*hidden_size)
        self.fc3 = nn.Linear(2*hidden_size, num_answers)
        self.decoder = decoder 

    def forward(self, question, lengths, image, amr):
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
        top_final_hidden = final_hidden[1]

        fwd_final_cell = final[1][0:final[1].size(0):2]
        bwd_final_cell = final[1][1:final[1].size(0):2]
        final_cell = torch.cat([fwd_final_cell, bwd_final_cell], dim=2)  # [num_layers, batch, 2*dim]

        q_embed = self.proj_q(top_final_hidden)
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

        encoder_hidden = output
        encoder_final = (final_hidden, final_cell)
        amr_probs, _ = self.decoder(amr, encoder_hidden,
                                   encoder_final, lengths,  
                                   teacher_forcing_ratio=1.0)

        amr_tokens = torch.max(amr_probs.transpose(2,1), dim=2)[1]
        amr_tokens = amr_tokens.detach()

        return pred, amr_probs, amr_tokens

def run_epoch(model, vqa_dataloader, question_lang, answer_lang, criterion, optim):
    total_loss = 0
    for idx, batch in enumerate(vqa_dataloader):
        img = batch["image"]
        question_tensor, lengths = batch_text(batch['question'], question_lang)
        out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, question_tensor)
        answer_tensors = batch_answers(batch["answer"], answer_lang)
        assert max(answer_tensors) < 1000
        loss = criterion(out, answer_tensors)
        loss += criterion(amr_probs, question_tensor)
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
            question_tensor, lengths = batch_text(batch['question'], question_lang)
            img = batch["image"]
            question_tensor, lengths = batch_text(batch['question'], question_lang)
            with torch.no_grad():
                out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, question_tensor)
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
        question_tensor, lengths = batch_text(batch['question'], question_lang)
        print(f'Example {idx}')
        print("question: ", batch["question"])
        print("answer: ", batch["answer"])
        print("amr: ", batch["amr"])
        img = batch["image"]
        question_tensor, lengths = batch_text(batch['question'], question_lang)
        with torch.no_grad():
            out, amr_probs, amr_tokens = model.forward(question_tensor, lengths, img, question_tensor)
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
            #val_accuracy = eval_accuracy(model, val_dataloader, question_lang, answer_lang, "/tmp/eval.txt")
            #print("val_accuracy", val_accuracy)
            #writer.add_scalar("val_accuracy", val_accuracy, epoch)

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

def make_model(question_lang, num_answers):
    resnet = torch.hub.load('pytorch/vision:v0.4.2', 'resnet50', pretrained=True) 

    num_layers = 2
    emb_size = 200
    dropout = 0.5
    hidden_size = cfg["hidden_size"]
    vocab_size = question_lang.n_words
    attention1 = BahdanauAttention(hidden_size)
    decoder = Decoder(emb_size, hidden_size, Softmax(hidden_size, vocab_size), 
                      nn.Embedding(vocab_size, emb_size),
                      attention1, num_layers=num_layers, dropout=dropout)
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
    train_df = make_df(train_json, 'abstract_v002_train2015_annotations.json', 'train_amr.txt')
    val_df = make_df(val_json, 'abstract_v002_val2015_annotations.json', 'val_amr.txt')
    
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
