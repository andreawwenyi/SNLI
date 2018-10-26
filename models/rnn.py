import torch
class RNN(torch.nn.Module):
    def __init__(self, dict_args, pre_trained_emb):
        # RNN Accepts the following hyperparams:
        # emb_size: Embedding Size
        # hidden_size: Hidden Size of layer in RNN
        # num_layers: number of layers in RNN
        # num_classes: number of output classes
        # vocab_size: vocabulary size
        super(RNN, self).__init__()
        
        self.device = torch.device(dict_args['device'])
        self.num_layers, self.hidden_size= dict_args['num_layers'], dict_args['hidden_size']
        self.emb_size = pre_trained_emb.size()[1]
        self.embedding = torch.nn.Embedding.from_pretrained(pre_trained_emb, freeze = True).to(self.device)
        self.bigru = torch.nn.GRU(self.emb_size, self.hidden_size, self.num_layers, batch_first=True,
                                  bidirectional=True).to(self.device)
        self.linear1 = torch.nn.Linear(4*self.hidden_size, self.hidden_size).to(self.device)
        self.relu =  torch.nn.ReLU().to(self.device)
        self.linear2 = torch.nn.Linear(self.hidden_size, dict_args['num_classes']).to(self.device)
        self.dropout = torch.nn.Dropout(dict_args['dropout'])
        
    def init_hidden(self, batch_size):
        # Function initializes the activation of recurrent neural net at timestep 0
        # Needs to be in format (num_layers, batch_size, hidden_size)
        hidden = torch.randn(2*self.num_layers, batch_size, self.hidden_size).to(self.device)

        return hidden

    def forward(self, prem, prem_len, hyp, hyp_len):
        
        
        batch_size, prem_seq_len = prem.size()
        _, hyp_seq_len = hyp.size()
        
        #sort
        prem_len_new, prem_perm_index = prem_len.sort(0, descending = True)
        hyp_len_new, hyp_perm_index = hyp_len.sort(0, descending = True)
        prem = prem[prem_perm_index]; hyp = hyp[hyp_perm_index]
        
        # get embedding of characters
        prem_embed = self.embedding(prem)
        hyp_embed = self.embedding(hyp)
        
        # pack padded sequence (pad shorter sequences, and pytorch want the sequence length in descending order. 
        # First element in batch should be the longest seqeunce.)
        
        packed_prem = torch.nn.utils.rnn.pack_padded_sequence(prem_embed, list(prem_len_new.data), batch_first=True)
        packed_hyp = torch.nn.utils.rnn.pack_padded_sequence(hyp_embed, list(hyp_len_new.data), batch_first=True)
        
        
        #init hidden layers for both premises and hypothese
        self.hidden_prem = self.init_hidden(batch_size)
        self.hidden_hyp = self.init_hidden(batch_size)
        
        #pass to bidirectional gru
        _, self.hidden_prem = self.bigru(packed_prem, self.hidden_prem)
        _, self.hidden_hyp = self.bigru(packed_hyp, self.hidden_hyp)
        
        #unsort
        _, prem_restore_index = prem_perm_index.sort(0)
        _, hyp_restore_index = hyp_perm_index.sort(0)
        
        #last hidden state
        prem_encode = torch.cat((self.hidden_prem[0], self.hidden_prem[1]), dim = 1)[prem_restore_index]
        hyp_encode = torch.cat((self.hidden_hyp[0], self.hidden_hyp[1]), dim = 1)[hyp_restore_index]
        
        #concat hypothesis and premises result
        out = torch.cat((prem_encode, hyp_encode), dim = 1)
        #size batch_size*(4*hidden_size)
        
        #pass to 2 FC layer
        out = self.linear1(out) #output size: batch_size*hidden_size
        out = self.dropout(out)
        out = self.relu(out) #output size: batch_size*hidden_size
        logits = self.linear2(out) #output batch_size * num_class

        return logits