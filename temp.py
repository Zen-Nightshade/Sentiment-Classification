from utlis import AttentiveRNN,  MultiplicativeAttentions
from transformers import BertTokenizer
import torch
import os

extracted_path = "Saved Models/Extracted Models"
bert_model = "prajjwal1/bert-mini"
tokenizer = BertTokenizer.from_pretrained(bert_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


attention = MultiplicativeAttentions(256, scoring_function="dot", bidirectional= False)
            
# Build model
model = AttentiveRNN(256, tokenizer.vocab_size, 512, attention=attention, bidirectional=False)

model.to(device)

try:
    model.load_state_dict(torch.load("Saved Models/Extracted Models/rnn_model_1.pth", map_location=device))
except RuntimeError as e:
    print("Attempt Failed\n\n")
    print(e)