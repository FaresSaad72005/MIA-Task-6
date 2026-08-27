
from . import split_data as split
from . import caption_preprocessing as caption
from . import imgfeature_extraction as imgex
from . import attention_model as model
from . import img_cap

import os
import pandas as pd
# from sklearn.model_selection import GroupShuffleSplit
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
DATASET_DIR = PROJECT_ROOT/"data"
# DATASET_DIR = "/mnt/e/linux_projects/MIA/task6/task6_2/data"
df = pd.read_csv(os.path.join(DATASET_DIR, "captions.txt"))
df["caption"] = df["caption"].astype(str).str.strip()

# ----- Data Split (.7 / .15 / .15) -----
train_df , val_df, test_df = split.split_data(df)

# Save datasets
train_df.to_csv(PROJECT_ROOT/"splits_vocab"/'train.csv', index=False)
# train_df.to_csv("/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/train.csv", index=False)
val_df.to_csv(PROJECT_ROOT/"splits_vocab"/'val.csv', index=False)
# val_df.to_csv("/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/val.csv", index=False)
test_df.to_csv(PROJECT_ROOT/"splits_vocab"/'test.csv', index=False)
# test_df.to_csv("/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/test.csv", index=False)



# ----- Caption Preprocessing -----
vocab = caption.Vocabulary(freq_threshold=2)
vocab.build_vocabulary(train_df["caption"].tolist())
print(f"Vocabulary Size: {len(vocab)} unique tokens")

# Save Vocab
torch.save(vocab, PROJECT_ROOT/"splits_vocab"/'vocab.pt')
# torch.save(vocab, "/mnt/e/linux_projects/MIA/task6/task6_2/splits_vocab/vocab.pt")
print("Saved vocabulary to vocab.pt")


# ----- Data preparing -----
# Image Transforms
transform = transforms.Compose( [ transforms.Resize((224, 224)), transforms.ToTensor(),
                                  transforms.Normalize( mean=[0.485, 0.456, 0.406],  # ImageNet standard values
                                                        std=[0.229, 0.224, 0.225]) ])

# Pair each transformed image to its numericalized caption
train_dataset = img_cap.Flickr8kDataset( df=train_df, images_dir=os.path.join(DATASET_DIR, "Images"), 
                                         vocab=vocab, transform=transform,)

pad_idx = vocab.stoi["<PAD>"]
collate_fn = caption.CaptionCollate(pad_idx=pad_idx)

train_loader = DataLoader( dataset=train_dataset, batch_size=32, 
                          shuffle=True, num_workers=2, collate_fn=collate_fn)







# ----- training loop -----
EMBED_SIZE = 256
HIDDEN_SIZE = 512
ATTENTION_DIM = 256
VOCAB_SIZE = len(vocab)
LEARNING_RATE = 0.0003
NUM_EPOCHS = 10

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = model.EncoderDecoderAttention( embed_size=EMBED_SIZE, hidden_size=HIDDEN_SIZE,
                                      vocab_size=VOCAB_SIZE, attention_dim=ATTENTION_DIM).to(DEVICE)

pad_idx = vocab.stoi["<PAD>"]
criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

images_directory = os.path.join(DATASET_DIR, "Images")


for epoch in range(NUM_EPOCHS):
    model.train()  # Ensure model is in train mode at start of epoch
    total_loss = 0.0

    for idx, (images, captions) in enumerate(train_loader):
        images = images.to(DEVICE)
        captions = captions.to(DEVICE)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        # Pass all tokens except the last to decoder
        outputs = model(images, captions[:, :-1])

        # Targets are shifted by 1 token
        targets = captions[:, 1:].contiguous().view(-1)
        outputs = outputs.contiguous().view(-1, outputs.size(-1))

        loss = criterion(outputs, targets)

        # Backward pass & optimize
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

        if (idx + 1) % 100 == 0:
            print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Step [{idx+1}/{len(train_loader)}], Step Loss: {loss.item():.4f}")

    # Calculate average loss after FULL epoch completes
    avg_loss = total_loss / len(train_loader)
    print(f"\n==> Epoch [{epoch+1}/{NUM_EPOCHS}] Average Loss: {avg_loss:.4f}")




# Define save path
SAVE_PATH = PROJECT_ROOT/"weights"/'train_attention_enc_dec1.pth'
# SAVE_PATH = "/mnt/e/linux_projects/MIA/task6/task6_2/weights/train_attention_enc_dec1.pth"

# Save state dictionary
torch.save(model.state_dict(), SAVE_PATH)
print(f"Model weights saved successfully to {SAVE_PATH}")