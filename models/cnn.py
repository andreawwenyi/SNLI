from torch import nn
import torch.nn.functional as F
import torch

class CNN(nn.Module):
    def __init__(self, dict_args, pre_trained_emb):

        super(CNN, self).__init__()
        self.device = torch.device(dict_args['device'])
        self.hidden_size, self.kernel_size = dict_args['hidden_size'], dict_args['kernel_size']
        self.emb_size = pre_trained_emb.size()[1]
        
        self.embedding = nn.Embedding.from_pretrained(pre_trained_emb, freeze = True).to(self.device)
        self.conv1 = nn.Conv1d(self.emb_size, self.hidden_size, kernel_size=self.kernel_size, padding=int((self.kernel_size-1)/2)).to(self.device)
        self.conv2 = nn.Conv1d(self.hidden_size, self.hidden_size, kernel_size=self.kernel_size, padding=int((self.kernel_size-1)/2)).to(self.device)
        self.linear1 = nn.Linear(2*self.hidden_size, self.hidden_size).to(self.device)
        self.linear2 = nn.Linear(self.hidden_size, dict_args['num_classes']).to(self.device)
        self.dropout = nn.Dropout(dict_args['dropout']).to(self.device)
        
    def forward(self, prem, prem_len, hyp, hyp_len):
        batch_size, prem_seq_len = prem.size()
        _, hyp_seq_len = hyp.size()

        prem_embed = self.embedding(prem)
        hyp_embed = self.embedding(hyp)
        
        #conv1
        prem_hidden = self.conv1(prem_embed.transpose(1,2)).transpose(1,2)
        prem_hidden = self.dropout(prem_hidden)
        prem_hidden = F.relu(prem_hidden.contiguous().view(-1, self.hidden_size)).view(batch_size, prem_seq_len, self.hidden_size)

        #conv2
        prem_hidden = self.conv2(prem_hidden.transpose(1,2)).transpose(1,2)
        prem_hidden = self.dropout(prem_hidden)
        prem_hidden = F.relu(prem_hidden.contiguous().view(-1, self.hidden_size)).view(batch_size, prem_seq_len, self.hidden_size)
        
        #maxpool
        prem_hidden, _  = prem_hidden.max(dim=1)
        
        #conv1
        hyp_hidden = self.conv1(hyp_embed.transpose(1,2)).transpose(1,2)
        hyp_hidden = self.dropout(hyp_hidden)
        hyp_hidden = F.relu(hyp_hidden.contiguous().view(-1, self.hidden_size)).view(batch_size, hyp_seq_len, self.hidden_size)
        
        #conv2
        hyp_hidden = self.conv2(hyp_hidden.transpose(1,2)).transpose(1,2)
        hyp_hidden = self.dropout(hyp_hidden)
        hyp_hidden = F.relu(hyp_hidden.contiguous().view(-1, self.hidden_size)).view(batch_size, hyp_seq_len, self.hidden_size)
        
        #maxpool
        hyp_hidden, _  = hyp_hidden.max(dim=1)
        
        #concat premises and hypothesis output
        out = torch.cat((prem_hidden, hyp_hidden), dim = 1)
        
        #2 FC layer
        out = self.linear1(out)
        out = F.relu(out)
        logits = self.linear2(out)
        return logits