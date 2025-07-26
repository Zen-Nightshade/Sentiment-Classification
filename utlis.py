import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
import matplotlib.pyplot as plt 
import seaborn as sns
import os
import zipfile


class AdditiveAttention(nn.Module):
    def __init__(self, hidden_state_dim, bidirectional = False):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        
        if bidirectional:
            self.input_dim = self.hidden_state_dim * 2
        else:
            self.input_dim = self.hidden_state_dim 
            
        self.U = nn.Linear(self.input_dim, self.hidden_state_dim)
        self.W = nn.Linear(self.input_dim, self.hidden_state_dim)
        self.V = nn.Linear(self.hidden_state_dim, 1)
    
    def forward(self, encoder_outputs, prev_hidden_state, attention_mask = None):
        # encoder_outputs: (batch, seq_len, input_dim)
        # prev_hidden_state: (directions, batch, hidden_state_dim)
        
        prev_hidden_state = prev_hidden_state.permute(1, 0, 2).contiguous()                     # (batch, directions, hidden_state_dim)
        prev_hidden_state = prev_hidden_state.view(prev_hidden_state.size(0), -1)               # (batch, input_dim)
        
        previous_state = prev_hidden_state.unsqueeze(1)                                         # (batch, 1, hidden_state_dim)
        Wa = self.W(encoder_outputs)                                                            # (batch, seq_len, hidden_state_dim)
        Ua = self.U(previous_state)                                                             # (batch, 1, hidden_state_dim) 
        energy = F.tanh(Wa+Ua)                                                                  # (batch, seq_len, hidden_state_dim)
        # shapes are broadcast-compatible, thanks to dimension 1 in Ua
        Score = self.V(energy)                                                                  # (batch, seq_len, 1)
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)                                                 # (batch, seq_len, 1)
            Score = Score.masked_fill(mask == 0, float('-inf'))                                 # (batch, seq_len, 1)
        
        attention_weights = F.softmax(Score, dim =1)                                            # (batch, seq_len, 1)
        
        context = (attention_weights * encoder_outputs).sum(dim = 1)                            #(batch, input_dim)
        # shapes are broadcast-compatible, thanks to dimension 1 in attention_weights
        return context, attention_weights
    
class MultiplicativeAttentions(nn.Module):
    def __init__(self, hidden_state_dim, bidirectional = False, scoring_function = "general"):
        super().__init__()
        self.hidden_state_dim = hidden_state_dim
        if bidirectional:
            self.input_dim = self.hidden_state_dim * 2
        else:
            self.input_dim = self.hidden_state_dim
        self.general = nn.Linear(self.input_dim, self.input_dim)
        self.concat = nn.Linear(self.input_dim * 2, self.hidden_state_dim)
        self.concate_V = nn.Linear(self.hidden_state_dim, 1)
        if scoring_function.lower() not in ["dot", "general", "concat"]:
            raise ValueError(f"Unsupported scoring function: {scoring_function}. Choose from 'dot', 'general', or 'concat'.")
        else:
            self.function = scoring_function.lower()
        
    def forward(self, encoder_outputs, prev_hidden_state, attention_mask = None):
        # encoder_outputs: (batch, seq_len, input_dim)
        # prev_hidden_state: (directions, batch, hidden_state_dim)

        prev_hidden_state = prev_hidden_state.permute(1, 0, 2).contiguous()                     # (batch, directions, hidden_state_dim)
        prev_hidden_state = prev_hidden_state.view(prev_hidden_state.size(0), -1)               # (batch, input_dim)
        batch, seq_len, _ = encoder_outputs.shape
        previous_state = prev_hidden_state.unsqueeze(1).expand(-1, seq_len, -1)                 # (batch, seq_len, input_dim)
        match self.function:
            case "dot":
                Score = torch.bmm(encoder_outputs, prev_hidden_state.unsqueeze(2))              # (batch, seq_len, 1)
            case "general":
                projected = self.general(encoder_outputs)                                       # (batch, seq_len, input_dim)
                Score = torch.bmm(projected, prev_hidden_state.unsqueeze(2))                    # (batch, seq_len, 1)
            case "concat":
                concated_inputs = torch.cat((encoder_outputs, previous_state), dim =2)          # (batch, seq_len, 2*input_dim)
                energy = F.tanh(self.concat(concated_inputs))                                   # (batch, seq_len, hidden_state_dim)
                Score = self.concate_V(energy)                                                  # (batch, seq_len, 1)
        
        if attention_mask is not None:
            mask = attention_mask.unsqueeze(-1)                                                 # (batch, seq_len, 1)
            Score = Score.masked_fill(mask == 0, float('-inf'))                                 # (batch, seq_len, 1)
            
        attention_weights = F.softmax(Score, dim = 1)                                           # (batch, seq_len, 1)

        context = (attention_weights * encoder_outputs).sum(dim=1)                              # (batch, input_dim)
        return context, attention_weights
    
class AttentiveRNN(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embed_dim, attention, bidirectional = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        if bidirectional:
            self.output_dim = self.hidden_dim * 2
        else:
            self.output_dim = self.hidden_dim
        self.rnn = nn.RNN(embed_dim, hidden_dim, batch_first=True, bidirectional = bidirectional)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = attention
        self.output = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, attention_mask = None):                                                                      # x: (batch, seq_len)
        x = self.embedding(x)
        encoder_outputs, last_hidden_state = self.rnn(x)
        # encoder_outputs: (batch, seq_len, output_dim)
        # last_hidden_state: (directions, batch, hidden_dim)
        batch, seq_len, _ = encoder_outputs.shape
        if self.attention is not None:
            context, attn_weights = self.attention(encoder_outputs, last_hidden_state, attention_mask)
            # context: (batch, output_dim)
            # attn_weights: (batch, seq_len, 1)
            context = self.dropout(context)
        else:
            context = last_hidden_state.permute(1, 0, 2).contiguous()                     # (batch, directions, hidden_state_dim)
            context = context.view(context.size(0), -1)                                   # (batch, output_dim)
            attn_weights = torch.full((batch, seq_len, 1), 1.0 / seq_len, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
            
        out = self.output(context)
        return out.squeeze(1), attn_weights.squeeze(2)
    
class AttentiveLSTM(nn.Module):
    def __init__(self, hidden_dim, vocab_size, embed_dim, attention, bidirectional = False):
        super().__init__()
        self.hidden_dim = hidden_dim
        if bidirectional:
            self.output_dim = self.hidden_dim * 2
        else:
            self.output_dim = self.hidden_dim
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional = bidirectional)
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.attention = attention
        self.output = nn.Linear(self.output_dim, 1)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, attention_mask):                                                                      # x: (batch, seq_len)
        x = self.embedding(x)
        encoder_outputs, (last_hidden_state, last_cell_state) = self.rnn(x)
        # encoder_outputs: (batch, seq_len, output_dim)
        # last_hidden_state: (directions, batch, hidden_dim)
        batch, seq_len, _ = encoder_outputs.shape
        if self.attention is not None:
            context, attn_weights = self.attention(encoder_outputs, last_hidden_state, attention_mask)
            # context: (batch, output_dim)
            # attn_weights: (batch, seq_len, 1)
            context = self.dropout(context)
        else:
            context = last_hidden_state.permute(1, 0, 2).contiguous()                     # (batch, directions, hidden_state_dim)
            context = context.view(context.size(0), -1)                                   # (batch, output_dim)
            attn_weights = torch.full((batch, seq_len, 1), 1.0 / seq_len, device=encoder_outputs.device, dtype=encoder_outputs.dtype)
            
        out = self.output(context)
        return out.squeeze(1), attn_weights.squeeze(2)
    

def visualize_models(models, text, device, model_names=None, bert_model='bert-base-uncased'):
    tokenizer = BertTokenizer.from_pretrained(bert_model)

    # Tokenize input
    encoding = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=512,
        return_tensors='pt'
    )
    input_ids = encoding["input_ids"].to(device)
    attention_mask = encoding["attention_mask"].to(device)

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0])
    seq_len = attention_mask[0].sum().item()
    tokens = tokens[:seq_len]

    if model_names is None:
        model_names = [f"Model {i+1}" for i in range(len(models))]

    heatmap_data = []
    y_labels = []

    for model, base_name in zip(models, model_names):
        model.eval()
        with torch.no_grad():
            outputs = model(input_ids, attention_mask)

            if not isinstance(outputs, tuple):
                raise ValueError("Model must return (logits, attention_weights) as output.")

            logits, attention_weights = outputs
            attention_weights = attention_weights[0].squeeze(-1)[:seq_len].cpu().numpy()
            prediction = torch.sigmoid(logits).item()
            label = 'p' if prediction > 0.5 else 'n'

            heatmap_data.append(attention_weights)
            y_labels.append(f"{base_name} - {label}")

    # Plot heatmap
    plt.figure(figsize=(min(24, seq_len), len(models) * 0.6 + 2))
    sns.heatmap(np.array(heatmap_data), xticklabels=tokens, yticklabels=y_labels, cmap='YlGnBu', cbar=True)
    plt.title("Attention Heatmap across Models")
    plt.xlabel("Input Tokens")
    plt.ylabel("Models (p=positive, n=negative)")
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()



def Zip_extractor(zip_dir, extract_dir):
    for file in os.listdir(zip_dir):
        if file.endswith(".zip"):
            zipped_file = os.path.join(zip_dir, file)
                        
            with zipfile.ZipFile(zipped_file, 'r') as zip_ref:
                zip_ref.extractall(extract_dir)