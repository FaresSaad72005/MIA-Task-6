import collections
import re
import torch
from torch.nn.utils.rnn import pad_sequence


class Vocabulary:

    def __init__(self, freq_threshold=2):
        self.freq_threshold = freq_threshold

        # Special tokens
        self.pad_token = "<PAD>"
        self.sos_token = "<SOS>" # Start of Sentence
        self.eos_token = "<EOS>" # End of Sentence
        self.unk_token = "<UNK>" # unknown words (OOV)

        self.itos = { 0: self.pad_token,
                     1: self.sos_token,
                     2: self.eos_token,
                     3: self.unk_token}
        
        self.stoi = {v: k for k, v in self.itos.items()}

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenize(text):
        # Makes all text lowercase and get rid off non-alphanumeric characters
        text = text.lower()
        tokens = re.findall(r"\b\w+\b", text)
        return tokens

    def build_vocabulary(self, sentence_list):
        frequencies = collections.Counter()
        idx = len(self.itos)

        # Count frequencies
        for sentence in sentence_list:
            tokens = self.tokenize(sentence)
            frequencies.update(tokens)

        # Add words meeting threshold
        for word, count in frequencies.items():
            if count >= self.freq_threshold:
                self.stoi[word] = idx
                self.itos[idx] = word
                idx += 1

    def numericalize(self, text):
        # Converts text to numbers
        tokens = self.tokenize(text)
        numericalized = [self.stoi[self.sos_token]]

        for token in tokens:
            numericalized.append(self.stoi.get(token, self.stoi[self.unk_token]))

        numericalized.append(self.stoi[self.eos_token])
        return numericalized



# add <PAD> token to make sure that all sequences will have the same length
class CaptionCollate:

    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        # batch contains tuples of (image_tensor, numericalized_caption_tensor)
        images = [item[0] for item in batch]
        captions = [item[1] for item in batch]

        images = torch.stack(images, dim=0)

        # Pad sequence along length dimension with <PAD> token ID
        padded_captions = pad_sequence(captions, batch_first=True, padding_value=self.pad_idx)

        return images, padded_captions