import torch
import torch.nn as nn 

class RNN(nn.Module):
    
    def __init__(self,
                 vocab_size,
                 lstm_size=2048,
                 embedding_size = 64,
                 num_layers=1,
                 batch_size=32):
        
        super().__init__()

        self.lstm_size = lstm_size
        self.embedding_size = embedding_size
        self.num_layers = num_layers
        self.vocab_size = vocab_size
        self.batch_size = batch_size

        self.embedding = nn.Embedding(self.vocab_size,self.embedding_size)
        self.lstm      = nn.LSTM(input_size = self.embedding_size,
                                 hidden_size = lstm_size,
                                 num_layers = num_layers,
                                 batch_first = True)
        self.fc     = nn.Linear(lstm_size,self.vocab_size)

    def forward(self, x, prev_state):
        embed        = self.embedding(x)
        output,state = self.lstm(embed, prev_state)
        state        = tuple(stat.detach() for stat in state) 
        out          = output.reshape(-1,self.lstm_size)
        out          = self.fc(out)

        return out, state
    
    def init_state(self):

        weight = next(self.parameters()).data

        if torch.cuda.is_available():
            hidden = (torch.zeros(self.num_layers, self.batch_size, self.lstm_size).cuda(),
                      torch.zeros(self.num_layers, self.batch_size, self.lstm_size).cuda())
        else:
            hidden = (torch.zeros(self.num_layers, self.batch_size, self.lstm_size),
                      torch.zeros(self.num_layers, self.batch_size, self.lstm_size))
        return hidden
