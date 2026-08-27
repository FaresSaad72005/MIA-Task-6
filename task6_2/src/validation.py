from . import caption_preprocessing as caption
from . import imgfeature_extraction as imgex
from . import attention_model as model
from . import img_cap
from . import cap_generation as cap_gen
from . import bleu

import os
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim

from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# ----- Load Data -----
val_df = pd.read_csv(PROJECT_ROOT/"splits_vocab"/"val.csv")
# val_df = pd.read_csv('/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/val.csv')
print(f'{val_df} is loaded successfully')

# ----- Load Vocab -----
vocab = torch.load(PROJECT_ROOT/"splits_vocab"/"vocab.pt", weights_only=False)
# vocab = torch.load("/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/vocab.pt", weights_only=False)
print(f'{vocab} is loaded successfully')


# DataLoader setup using the CaptionCollate from earlier
pad_idx = vocab.stoi["<PAD>"]
collate_fn = caption.CaptionCollate(pad_idx=pad_idx)


# ----- Data preparing -----
# Image Transforms
transform = transforms.Compose( [ transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize( mean=[0.485, 0.456, 0.406],  # ImageNet standard values
                                                        std=[0.229, 0.224, 0.225]) ])


DATASET_DIR = PROJECT_ROOT/"data"
# DATASET_DIR = "/mnt/e/linux_projects/MIA/task6/task6_2/data"
images_directory = os.path.join(DATASET_DIR, "Images")


#  ----- Creating DataLoader -----
val_dataset = img_cap.Flickr8kDataset( df=val_df, images_dir=images_directory, 
                                      vocab=vocab, transform=transform, )
val_loader = DataLoader(dataset=val_dataset, batch_size=32, 
                        shuffle=False, collate_fn=collate_fn)


#  ----- Loading Saved model weight -----
EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 256
VOCAB_SIZE = len(vocab)


SAVE_PATH = PROJECT_ROOT/"weights"/"train_attention_enc_dec1.pth"
# SAVE_PATH = "/mnt/e/linux_projects/MIA/task6/task6_2/weights/train_attention_enc_dec1.pth"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.EncoderDecoderAttention( embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                                       vocab_size=VOCAB_SIZE, attention_dim=ATTENTION_DIM).to(DEVICE)

model.load_state_dict(torch.load(SAVE_PATH, map_location=DEVICE, weights_only=True))

model.eval()
print("Attention model loaded successfully and ready for inference!")





# BLEU scores of Validation
print("\nCalculating Final Validation BLEU Scores...")
bleu_scores = bleu.bleu(model, val_df, vocab, transform, images_directory, DEVICE)

for metric, score in bleu_scores.items():
    print(f"{metric}: {score:.2f}")