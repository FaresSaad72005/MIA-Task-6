# import collections
# import os
import torch
from nltk.translate.bleu_score import SmoothingFunction, corpus_bleu
from PIL import Image



def generate_caption_attention(model, image_path, transform, vocab, device, max_length=20):
    model.eval()
    raw_img = Image.open(image_path).convert("RGB")
    img_tensor = transform(raw_img).unsqueeze(0).to(device)

    predicted_tokens = []
    
    with torch.no_grad():
        encoder_features = model.encoder(img_tensor)
        h, c = model.decoder.init_hidden_state(encoder_features)
        
        word_id = vocab.stoi["<SOS>"]
        
        for _ in range(max_length):
            context, _ = model.decoder.attention(encoder_features, h)
            embed = model.decoder.embedding(torch.tensor([word_id]).to(device))
            
            lstm_input = torch.cat([embed, context], dim=1)
            h, c = model.decoder.decode_step(lstm_input, (h, c))
            
            output = model.decoder.fc(h)
            word_id = output.argmax(1).item()
            
            predicted_word = vocab.itos[word_id]
            if predicted_word == "<EOS>":
                break
                
            predicted_tokens.append(predicted_word)
            
    return " ".join(predicted_tokens)

