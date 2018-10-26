import torch
from torch.utils.data import Dataset
import numpy as np
class SNLIDataset(Dataset):
    """
    Class that represents a train/validation/test dataset that's readable for PyTorch
    Note that this class inherits torch.utils.data.Dataset
    """

    def __init__(self, dataset):
        """
        @Arg dataset: pandas dataframe. All the tokens should already be turned into indices. 
                      column 1 is premises; column 2 is hypothese; column 3 is target 
        @param prem_list: list of premises
        @param hyp_list: list of hypothesis
        @param target_list: list of targets

        """
        self.prem_list, self.hyp_list, self.target_list = dataset.iloc[:,0].tolist(), dataset.iloc[:,1].tolist(), dataset.iloc[:,2].tolist()
        assert (len(self.prem_list) == len(self.hyp_list) == len(self.target_list)) 
        

    def __len__(self):
        return len(self.target_list)

    def __getitem__(self, key):
        """
        Triggered when you call dataset[i]
        """
        return [self.prem_list[key], len(self.prem_list[key]), 
                self.hyp_list[key], len(self.hyp_list[key]), 
                self.target_list[key]]
    
def snli_collate_func(batch):
    """
    Customized function for DataLoader that dynamically pads the batch so that all
    data have the same length
    """
    prem_list = []
    hyp_list = []
    label_list = []
    prem_length_list = []
    hyp_length_list = []
    
    for datum in batch:
        label_list.append(datum[4])
        prem_length_list.append(datum[1])
        hyp_length_list.append(datum[3])
    max_prem_length = max(prem_length_list)
    max_hyp_length = max(hyp_length_list)
    
    # padding
    for datum in batch:
        padded_prem = np.pad(np.array(datum[0]), pad_width=((0,max_prem_length-datum[1])),
                                mode="constant", constant_values=0)
        prem_list.append(padded_prem)
        
        padded_hyp = np.pad(np.array(datum[2]), pad_width=((0,max_hyp_length-datum[3])),
                                mode="constant", constant_values=0)
        hyp_list.append(padded_hyp)
    
    return [torch.from_numpy(np.array(prem_list)), torch.LongTensor(prem_length_list), 
            torch.from_numpy(np.array(hyp_list)), torch.LongTensor(hyp_length_list), 
            torch.LongTensor(label_list)]
