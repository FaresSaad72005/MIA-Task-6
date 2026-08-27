import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

# prepares image-caption pairs
class Flickr8kDataset(Dataset):

    def __init__(self, df, images_dir, vocab, transform=None):
        self.df = df
        self.images_dir = images_dir
        self.vocab = vocab
        self.transform = transform

        self.images = self.df["image"].tolist()
        self.captions = self.df["caption"].tolist()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        caption = self.captions[index]
        img_id = self.images[index]
        img_path = os.path.join(self.images_dir, img_id)

        # Load & Transform Image
        image = Image.open(img_path).convert("RGB")
        if self.transform is not None:
            image = self.transform(image)

        # Numericalize Caption
        numericalized_caption = self.vocab.numericalize(caption)

        return image, torch.tensor(numericalized_caption)

