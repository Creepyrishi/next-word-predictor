from flask import Flask, request, jsonify, render_template
import torch
import torch.nn as nn
import json
import os
import numpy as np

# Model classes (copied from notebook)
class Normal_LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim=100, padding_idx=0)
        self.do = nn.Dropout(.1)
        self.lstm = nn.LSTM(input_size=100, hidden_size=150, num_layers=2, batch_first=True)
        self.layers = nn.Sequential(
            nn.Linear(in_features=150, out_features=200),
            nn.Dropout(.2),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=vocab_size)
        )
    def forward(self, x):
        x = self.emb(x)
        y = self.do(x)
        h, c = self.lstm(x)
        y = h[:, -1, :]
        y = self.layers(y)
        return y

class Bi_LSTM(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim=100, padding_idx=0)
        self.do = nn.Dropout(.1)
        self.lstm = nn.LSTM(input_size=100, hidden_size=150, num_layers=2, batch_first=True, bidirectional=True)
        self.layers = nn.Sequential(
            nn.Linear(in_features=150*2, out_features=200),
            nn.Dropout(.2),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=vocab_size)
        )
    def forward(self, x):
        x = self.emb(x)
        y = self.do(x)
        h, c = self.lstm(x)
        y = h[:, -1, :]
        y = self.layers(y)
        return y

class bi_GRU(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, embedding_dim=100, padding_idx=0)
        self.do = nn.Dropout(.1)
        self.gru = nn.GRU(input_size=100, hidden_size=150, num_layers=2, batch_first=True, bidirectional=True)
        self.layers = nn.Sequential(
            nn.Linear(in_features=2*150, out_features=200),
            nn.Dropout(.2),
            nn.ReLU(),
            nn.Linear(in_features=200, out_features=vocab_size)
        )
    def forward(self, x):
        x = self.emb(x)
        y = self.do(x)
        h, c = self.gru(x)
        y = h[:, -1, :]
        y = self.layers(y)
        return y

# Flask app
app = Flask(__name__)

# Load vocab
with open('../vocab.json', 'r') as f:
    w_to_i = json.load(f)
    i_to_w = {int(v): k for k, v in w_to_i.items()}

vocab_size = len(w_to_i) + 1
max_words = 30

def sentence_to_int(sentence):
    return [w_to_i.get(word, w_to_i['<UNK>']) for word in sentence.lower().split()]

def predict_next_words(model, text, num_words=3, device='cpu'):
    model.eval()
    x = text
    for _ in range(num_words):
        x_int = sentence_to_int(x)
        x_tensor = torch.tensor(x_int).unsqueeze(0).to(device)
        with torch.no_grad():
            y = model(x_tensor)
            next_word_idx = torch.argmax(y, dim=1).item()
            next_word = i_to_w.get(next_word_idx, '<UNK>')
            x += ' ' + next_word
    return x.split()[-num_words:]

# Load models
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
models = {}
model_files = {
    'LSTM': '../models/LSTM_best_model.pt',
    'BiLSTM': '../models/BiLSTM_best_model.pt',
    'BiGRU': '../models/BiGRU_best_model.pt',
}

for name, path in model_files.items():
    if name == 'LSTM':
        model = Normal_LSTM(vocab_size)
    elif name == 'BiLSTM':
        model = Bi_LSTM(vocab_size)
    elif name == 'BiGRU':
        model = bi_GRU(vocab_size)
    model.load_state_dict(torch.load(path, map_location=DEVICE))
    model.to(DEVICE)
    model.eval()
    models[name] = model

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    text = data.get('text', '')
    model_name = data.get('model', 'LSTM')
    model = models.get(model_name)
    if not model:
        return jsonify({'error': 'Invalid model name'}), 400
    next_words = predict_next_words(model, text, num_words=3, device=DEVICE)
    return jsonify({'next_words': next_words})

if __name__ == '__main__':
    app.run(debug=True) 