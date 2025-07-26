from utlis import Zip_extractor, visualize_models
from utlis import AttentiveRNN, AttentiveLSTM, MultiplicativeAttentions
from transformers import BertTokenizer
import torch
import os

extracted_path = "Saved Models/Extracted Models"
bert_model = "prajjwal1/bert-mini"
tokenizer = BertTokenizer.from_pretrained(bert_model)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Models = []
Model_names = []

# Zip_extractor(zip_dir="Saved Models/Zipped Models", extract_dir= extracted_path)

model_types = ["RNN", "LSTM"]
score_functions = [None, "dot", "general", "concat"]
directions = [False, True]  # False = uni, True = bi


for model_type in model_types:
    idx = 0  # Unique model index for file naming
    for direction in directions:
        for score_function in score_functions:
            # Build attention
            attention = None if score_function is None else MultiplicativeAttentions(256, scoring_function=score_function, bidirectional= direction)
            
            # Build model
            if model_type == "RNN":
                model = AttentiveRNN(256, tokenizer.vocab_size, 512, attention=attention, bidirectional=direction)
            else:
                model = AttentiveLSTM(256, tokenizer.vocab_size, 512, attention=attention, bidirectional=direction)
            
            model.to(device)

            # Load weights
            file_prefix = "rnn_model_" if model_type == "RNN" else "lstm_model_"
            saved_path = os.path.join(extracted_path, f"{file_prefix}{idx}.pth")
            try:
                model.load_state_dict(torch.load(saved_path, map_location=device))
            except RuntimeError as e:
                print(f"model_type: {model_type}")
                print(f"scoring_function: {score_function}")
                print(f"bidirection: {direction}\n\n")
                print(f"[Skipping] Mismatch while loading: {saved_path}\n\n")
                print(e)
                print("\n\n")
                continue

            # Append to lists
            Models.append(model)

            model_name = f"{'Bi-' if direction else ''}{model_type}_"
            model_name += score_function if score_function else "vanilla"
            Model_names.append(model_name)

            idx += 1  # increment unique ID for file naming


# text = "As a fan of the original animated series, The Last Airbender was a crushing disappointment The characters felt flat and unrecognizable, the dialogue was stiff, and the plot rushed through key moments with no emotional payoff The bending effects were underwhelming, and the lack of chemistry between the cast made everything feel forced It completely missed the charm, depth, and spirit of the source material, turning a rich world into a dull, joyless spectacle."
# text = "And and and and and and and and and and and and and and and and"
# text = "It was not a disappointment at all, totaly worth it and a good movie"
visualize_models(Models, text, device= device, model_names=Model_names, bert_model= bert_model)