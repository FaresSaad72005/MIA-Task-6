import torch
import pandas as pd
from PIL import Image
import torchvision.transforms as transforms

from . import attention_model as attention

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Paths
VOCAB_PATH = PROJECT_ROOT/"splits_vocab"/"vocab.pt"
# VOCAB_PATH = "/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/vocab.pt"
MODEL_PATH = PROJECT_ROOT/"weights"/"train_attention_enc_dec1.pth"
# MODEL_PATH = "/mnt/e/linux_projects/MIA/task6/task6_2/weights/train_attention_enc_dec1.pth"


# Device
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Model configuration
EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 256

import sys
from src import caption_preprocessing

# Redirect legacy unpickler references to the new package path
sys.modules['caption_preprocessing'] = caption_preprocessing

# Load vocabulary

vocab = torch.load( VOCAB_PATH, weights_only=False)
VOCAB_SIZE = len(vocab)
print(f"Vocabulary loaded successfully ({VOCAB_SIZE} words)")


# Load model

model = attention.EncoderDecoderAttention( embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE, 
                                           vocab_size=VOCAB_SIZE, attention_dim=ATTENTION_DIM).to(DEVICE)

model.load_state_dict( torch.load( MODEL_PATH, map_location=DEVICE, weights_only=True))
model.eval()
print("Attention model loaded successfully and ready for inference!")



# Image transformation
transform = transforms.Compose([ transforms.Resize((224, 224)), transforms.ToTensor(),
                                 transforms.Normalize( mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


# Caption generation

def deploy(image, max_length=20):
    model.eval()

    raw_img = image.convert("RGB")

    img_tensor = transform(raw_img).unsqueeze(0).to(DEVICE)

    predicted_tokens = []

    with torch.no_grad():

        encoder_features = model.encoder(img_tensor)

        h, c = model.decoder.init_hidden_state( encoder_features)

        word_id = vocab.stoi["<SOS>"]

        for _ in range(max_length):

            context, _ = model.decoder.attention( encoder_features, h )

            embed = model.decoder.embedding( torch.tensor([word_id]).to(DEVICE) )

            lstm_input = torch.cat( [embed, context], dim=1 )

            h, c = model.decoder.decode_step( lstm_input, (h, c) )

            output = model.decoder.fc(h)

            word_id = output.argmax(1).item()

            predicted_word = vocab.itos[word_id]

            if predicted_word == "<EOS>":
                break

            predicted_tokens.append(predicted_word)

    return " ".join(predicted_tokens)