import torch
from torch import no_grad
import torch.nn as nn
import torch.optim as optim
import random
import math
import os

# ----------------------------
# LSTM Seq2Seq for Bounding Box Prediction
# ----------------------------

class LSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, middle_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super(LSTMPredictor, self).__init__()

        self.fc_in = nn.Linear(input_dim, middle_dim, dtype=torch.float32)
        if num_layers == 1:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dtype=torch.float32)
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dtype=torch.float32)
        else:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
        self.fc_out = nn.Linear(hidden_dim, input_dim, dtype=torch.float32)  # predict offset (dx, dy, dw, dh)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        # src: (batch, seq_len, 4)
        # trg: (batch, seq_len, 4) - ground truth future sequence
        outputs = []

        batch_size, trg_size, _ = trg.size()
        # Encode
        _, (hidden, cell) = self.encoder(self.fc_in(src))

        # First input to decoder is the last frame of src
        decoder_input = trg[:, 0:1, :]  # shape (batch, 1, 4)


        for t in range(1, trg_size + 1):
            out, (hidden, cell) = self.decoder(self.fc_in(decoder_input), (hidden, cell))
            pred = self.fc_out(out)  # (batch, 1, 4)
            outputs.append(pred)

            # Decide if we use teacher forcing
            if t != trg_size:
                use_teacher = trg is not None and random.random() < teacher_forcing_ratio
                decoder_input = trg[:, t:t+1, :] if use_teacher else (pred + trg[:, t-1:t, :])

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 4)
        return outputs + trg
    
    @no_grad
    def inference(self, src, trg, num_steps=1):
        outputs = []
        # Encode
        _, (hidden, cell) = self.encoder(self.fc_in(src))

        # First input to decoder is the last frame of src
        decoder_input = trg[:, 0:1, :]  # shape (batch, 1, 4)


        for t in range(num_steps):
            out, (hidden, cell) = self.decoder(self.fc_in(decoder_input), (hidden, cell))
            pred = self.fc_out(out)  # (batch, 1, 4)
            decoder_input = pred + decoder_input
            outputs.append(decoder_input)

        outputs = torch.cat(outputs, dim=1)  # (batch, seq_len, 4)
        return outputs


    def train_one_epoch(self, dataloader, optimizer, criterion, teacher_forcing_ratio=0.5, device='cuda'):
        self.train()
        total_loss = 0

        for src, trg in dataloader:
            src = src.to(device)
            trg = trg.to(device)

            optimizer.zero_grad()
            output = self.forward(src, trg[:, :-1], teacher_forcing_ratio)

            loss = criterion(output, trg[:, 1:])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()

        return total_loss / len(dataloader)


    def evaluate(self, dataloader, criterion, device='cuda'):
        self.eval()
        total_loss = 0

        with torch.no_grad():
            for src, trg in dataloader:
                src = src.to(device)
                trg = trg.to(device)
                output = self.inference(src, trg, num_steps=trg.size(1) - 1)
                loss = criterion(output, trg[:, 1:])
                total_loss += loss.item()

        return total_loss / len(dataloader)
    
    def save_weight(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(self.state_dict(), path)

    def load_weight(self, path):
        self.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))

class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, middle_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        self.fc_in = nn.Linear(input_dim, middle_dim, dtype=torch.float32)
        if num_layers > 1:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True, dtype=torch.float32)
        else:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True, dtype=torch.float32)
        # self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True, dtype=torch.float32)
        self.attn_fc = nn.Linear(hidden_dim*2, middle_dim, dtype=torch.float32)
        self.attn = nn.MultiheadAttention(middle_dim, num_heads=4, batch_first=True, dtype=torch.float32)
        if num_layers > 1:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
        else:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dtype=torch.float32)
        # self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, dtype=torch.float32)
        self.fc_out = nn.Linear(hidden_dim, input_dim, dtype=torch.float32)

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size, _, _ = src.size()
        _, trg_size, _ = trg.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        # Project encoder outputs for attention
        proj_enc = torch.tanh(self.attn_fc(enc_out))  # (batch, seq, middle_dim)

        prev_box = trg[:, 0:1, :]
        outputs = []
        for t in range(1, trg_size + 1):
            inp = self.fc_in(prev_box)
            # built-in multi-head attent ion
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out  # residual
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            # pred = prev_box + delta  # residual update
            pred = trg[:, t-1:t, :] + delta
            outputs.append(pred)
            if t < trg_size:
                use_teacher = trg is not None and random.random() < teacher_forcing_ratio
                prev_box = trg[:, t:t+1, :] if use_teacher else pred
        return torch.cat(outputs, dim=1)

    @no_grad
    def inference(self, src, trg, num_steps=1):
        batch_size, _, _ = src.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        proj_enc = torch.tanh(self.attn_fc(enc_out))

        prev_box = trg[:, 0:1, :]
        outputs = []
        for _ in range(num_steps):
            inp = self.fc_in(prev_box)
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            pred = prev_box + delta
            outputs.append(pred)
            prev_box = pred
        return torch.cat(outputs, dim=1)
    
class ImprovedLSTMPredictor(nn.Module):
    def __init__(self, input_dim=4, middle_dim=16, hidden_dim=64, num_layers=2, dropout=0.2):
        super().__init__()
        # self.fc_in = nn.Linear(input_dim, middle_dim)
        self.fc_in = nn.Sequential(
            nn.Linear(input_dim, 16),
            nn.LeakyReLU(0.1),
            nn.Linear(16, middle_dim)
        )
        if num_layers > 1:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout, bidirectional=True)
        else:
            self.encoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.attn_fc = nn.Linear(hidden_dim*2, middle_dim)
        self.attn = nn.MultiheadAttention(middle_dim, num_heads=4, batch_first=True)
        if num_layers > 1:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True, dropout=dropout)
        else:
            self.decoder = nn.LSTM(middle_dim, hidden_dim, num_layers, batch_first=True)
        # self.fc_out = nn.Linear(hidden_dim, input_dim)
        self.fc_out =nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.LeakyReLU(0.1),
            nn.Linear(64, input_dim)
        )

    def forward(self, src, trg=None, teacher_forcing_ratio=0.5):
        batch_size, _, _ = src.size()
        _, trg_size, _ = trg.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        # Project encoder outputs for attention
        proj_enc = torch.tanh(self.attn_fc(enc_out))  # (batch, seq, middle_dim)

        prev_box = trg[:, 0:1, :]
        outputs = []
        for t in range(1, trg_size + 1):
            inp = self.fc_in(prev_box)
            # built-in multi-head attent ion
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out  # residual
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            # pred = prev_box + delta  # residual update
            pred = trg[:, t-1:t, :] + delta
            outputs.append(pred)
            if t < trg_size:
                use_teacher = trg is not None and random.random() < teacher_forcing_ratio
                prev_box = trg[:, t:t+1, :] if use_teacher else pred
        return torch.cat(outputs, dim=1)

    @no_grad
    def inference(self, src, trg, num_steps=1):
        batch_size, _, _ = src.size()
        enc_embed = self.fc_in(src)
        enc_out, (h, c) = self.encoder(enc_embed)
        # Combine bidirectional states
        h = h.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        c = c.view(self.encoder.num_layers, 2, batch_size, self.encoder.hidden_size).sum(dim=1)
        proj_enc = torch.tanh(self.attn_fc(enc_out))

        prev_box = trg[:, 0:1, :]
        outputs = []
        for _ in range(num_steps):
            inp = self.fc_in(prev_box)
            attn_out, _ = self.attn(inp, proj_enc, proj_enc)
            dec_input = inp + attn_out
            # out, _ = self.decoder(dec_input, (h, c))
            out, (h, c) = self.decoder(dec_input, (h, c))
            delta = self.fc_out(out)
            pred = prev_box + delta
            outputs.append(pred)
            prev_box = pred
        return torch.cat(outputs, dim=1)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                       # (1,L,D)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1)]
        
class MotionTransformer(nn.Module):
    def __init__(self, input_dim=4, d_model=128, nhead=8,
                 num_enc_layers=3, num_dec_layers=3,
                 dim_ff=512, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.in_fc = nn.Linear(input_dim, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead,
                                          num_encoder_layers=num_enc_layers,
                                          num_decoder_layers=num_dec_layers,
                                          dim_feedforward=dim_ff,
                                          dropout=dropout, batch_first=True)
        self.out_fc = nn.Linear(d_model, input_dim)  # predicts offset Δ

    def encode(self, src):
        src_emb = self.pos_enc(self.in_fc(src) * math.sqrt(self.d_model))
        memory = self.transformer.encoder(src_emb)
        return memory

    def decode_step(self, memory, dec_in):
        # dec_in: (B,t,4)
        dec_emb = self.pos_enc(self.in_fc(dec_in) * math.sqrt(self.d_model))
        tgt_mask = self._causal_mask(dec_emb.size(1), device=dec_emb.device)
        out = self.transformer.decoder(dec_emb, memory, tgt_mask=tgt_mask)
        pred_offset = self.out_fc(out[:, -1:, :])     # predict Δ for last step
        return pred_offset

    @staticmethod
    def _causal_mask(size, device):
        return torch.triu(torch.ones(size, size, device=device), diagonal=1).bool()

    # -------- training forward pass with teacher forcing --------
    def forward(self, src, trg, teacher_forcing_ratio=1.0):
        """
        src: (B,S,4) observed boxes
        trg: (B,T,4) future *absolute* boxes
        returns predicted *absolute* boxes for the whole target horizon
        """
        B, T, _ = trg.shape
        memory = self.encode(src)

        # start with the last observed box as initial decoder input
        prev_box = src[:, -1:, :]
        preds = []

        for t in range(T):
            offset = self.decode_step(memory, prev_box)
            # compute next box as prev_box + predicted offset
            next_box = prev_box[:, -1:, :] + offset
            preds.append(next_box)

            if t < T-1:
                # choose next decoder input: ground truth or model prediction
                use_teacher = (random.random() < teacher_forcing_ratio)
                next_in = trg[:, t:t+1, :] if use_teacher else next_box
                prev_box = torch.cat([prev_box, next_in], dim=1)

        return torch.cat(preds, dim=1)   # (B,T,4)

    # -------- inference rollout (no teacher forcing) --------
    @torch.no_grad()
    def generate(self, src, steps):
        memory = self.encode(src)
        prev_box = src[:, -1:, :]
        preds = []
        for _ in range(steps):
            offset = self.decode_step(memory, prev_box)
            next_box = prev_box[:, -1:, :] + offset
            preds.append(next_box)
            prev_box = torch.cat([prev_box, next_box], dim=1)
        return torch.cat(preds, dim=1)

device = 'cuda'
model = LSTMPredictor(middle_dim=64, hidden_dim=256, num_layers=1).to(device)
model.load_weight('lstm-base-m64-h256-wn-v2.pth')
# model = LSTMPredictor(middle_dim=32, hidden_dim=128, num_layers=2).to(device)
# model.load_state_dict(torch.load('best_lstm_model (1).pth', map_location='cuda', weights_only=True))

# torch.manual_seed(145)
# model = ImprovedLSTMPredictor().to(device)
# model.load_state_dict(torch.load('best_improved_lstm_model.pth', map_location='cuda', weights_only=True))

# torch.manual_seed(49)
# model = ImprovedLSTMPredictor(middle_dim=64, hidden_dim=128).to(device)
# model.load_state_dict(torch.load('best_improved_lstm_model-m64-h128_ft.pth', map_location='cuda', weights_only=True))

# model = ImprovedLSTMPredictor(middle_dim=64, hidden_dim=256, num_layers=1).to(device)
# model.load_state_dict(torch.load('best_improved_lstm_model-m64-h256_l1_ft3.pth', map_location='cuda', weights_only=True))

# model = ImprovedLSTMPredictor(middle_dim=64, hidden_dim=256, num_layers=1).to(device)
# model.load_state_dict(torch.load('best_improved_lstm_model-m64-h256-l1-base.pth', map_location='cuda', weights_only=True))
# model.load_state_dict(torch.load('best_improved_lstm_model-m64-h256-l1-ft.pth', map_location='cuda', weights_only=True))

# model = MotionTransformer(num_enc_layers=1, num_dec_layers=1, dim_ff=256).to(device)
# model.load_state_dict(torch.load('best_transformer-e1-d1-ff256.pth', map_location='cuda', weights_only=True))
