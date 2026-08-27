from . import imgfeature_extraction as imgex
import torch
import torch.nn as nn




class BahdanauAttention(nn.Module):
    def __init__(self, encoder_dim, decoder_dim, attention_dim):
        super(BahdanauAttention, self).__init__()
        self.encoder_att = nn.Linear(encoder_dim, attention_dim)
        self.decoder_att = nn.Linear(decoder_dim, attention_dim)
        self.full_att = nn.Linear(attention_dim, 1)
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax(dim=1)

    def forward(self, encoder_features, decoder_hidden):
        # encoder_features: (Batch, 49, 2048)
        # decoder_hidden: (Batch, decoder_dim)
        
        att1 = self.encoder_att(encoder_features)       # (Batch, 49, attention_dim)
        att2 = self.decoder_att(decoder_hidden)         # (Batch, attention_dim)
        
        # Add dimensions for broadcasting -> (Batch, 49, attention_dim)
        score = self.full_att(self.relu(att1 + att2.unsqueeze(1))) # (Batch, 49, 1)
        score = score.squeeze(2)                         # (Batch, 49)
        
        alpha = self.softmax(score)                      # (Batch, 49)
        context = (encoder_features * alpha.unsqueeze(2)).sum(dim=1) # (Batch, 2048)
        
        return context, alpha

    



class DecoderWithAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, encoder_dim=2048, attention_dim=256, dropout=0.3):
        super(DecoderWithAttention, self).__init__()
        
        self.vocab_size = vocab_size
        self.attention = BahdanauAttention(encoder_dim, hidden_size, attention_dim)
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(dropout)
        self.decode_step = nn.LSTMCell(embed_size + encoder_dim, hidden_size)
        
        # Initial hidden state projections from mean spatial features
        self.init_h = nn.Linear(encoder_dim, hidden_size)
        self.init_c = nn.Linear(encoder_dim, hidden_size)
        
        self.fc = nn.Linear(hidden_size, vocab_size)

    def init_hidden_state(self, encoder_features):
        mean_encoder_features = encoder_features.mean(dim=1)
        h = torch.tanh(self.init_h(mean_encoder_features))
        c = torch.tanh(self.init_c(mean_encoder_features))
        return h, c

    def forward(self, encoder_features, captions):
        # captions: (Batch, max_seq_len)
        batch_size = encoder_features.size(0)
        seq_len = captions.size(1)
        
        embeddings = self.embedding(captions)  # (Batch, seq_len, embed_size)
        h, c = self.init_hidden_state(encoder_features)
        
        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(encoder_features.device)
        
        for t in range(seq_len):
            context, _ = self.attention(encoder_features, h)
            
            # Concatenate token embedding with image context vector
            lstm_input = torch.cat([embeddings[:, t, :], context], dim=1)
            h, c = self.decode_step(lstm_input, (h, c))
            
            output = self.fc(self.dropout(h))
            outputs[:, t, :] = output
            
        return outputs



class EncoderDecoderAttention(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, attention_dim=256):
        super(EncoderDecoderAttention, self).__init__()
        self.encoder = imgex.ResNetEncoder(embed_size=embed_size)
        self.decoder = DecoderWithAttention( embed_size=embed_size, hidden_size=hidden_size,
                                             vocab_size=vocab_size, attention_dim=attention_dim)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs