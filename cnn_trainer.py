import tqdm
import torch
import torch.nn.functional as F
from rnn_trainer import rnn_trainer
from models.cnn import CNN
from SNLI_DataLoader import SNLIDataset, snli_collate_func

class cnn_trainer(rnn_trainer):
    
    def __init__(self, train_data, val_data, pre_trained_emb, args):
        super(rnn_trainer, self).__init__()
        
        self.device = torch.device(args['device'])
        
        #data
        self.train_data = train_data
        self.val_data = val_data
        self.batch_size = args['batch_size']
        
        #init model
        model_args = {'hidden_size': args['hidden_size'], 
                     'num_classes': args['num_classes'],
                      'kernel_size': args['kernel_size'],
                     'dropout': args['dropout'],
                     'device': self.device
                     }
        self.model = CNN(model_args, pre_trained_emb)
        
        #training-level parameter
        self.lr = args['learning_rate']
        if args['optim'] == 'adam':
            self.optim = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.num_epochs = args['num_epochs']
