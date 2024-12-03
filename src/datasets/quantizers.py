import torch
import warnings
import numpy as np
from tqdm import tqdm

from torch.utils.data import DataLoader
from src.models.quantformer import Model
from sklearn.preprocessing import KBinsDiscretizer

class Quantizer():
    
    def __init__(self, dataset, num_levels, strategy, patcher):
        
        """
        Parameters
        -----------
        dataset : torch.utils.data.Dataset
            is the dataset to use to compute the quantization levels. It must have a responses attribute. 
            Responses is a dict that must have the following structure:
            {
                'response_name': {
                    'trace': np.ndarray (shape [n_samples x trace_len]),
                    ...
            }
        num_levels : int
            is the number of levels to use for the quantization
        strategy : str
            is the strategy to use for the quantization. It can be 'uniform' or 'quantile'
        patcher : Patcher
            is the patcher to use to get the patches
        """
        
        ### Create quantizer i.e. create the quantization levels
        
        ## Get traces
        traces = [torch.tensor(dataset.responses[k]['trace']) for k in dataset.responses.keys()]
        traces = torch.cat(traces, dim=0)
        
        ## Filter traces (Average Filter)
        w = patcher.patch_len // 2
        filtered_traces = torch.zeros_like(traces)
        for t in range(filtered_traces.shape[0]):
            filtered_traces[max(t-w, 0):t+w] = traces[max(t-w, 0):t+w].mean(axis=0)
        
        # Try to quantize until we get the desired number of levels (some levels may not be used)
        num_classes = num_levels
        num_real_levels = 0
        
        while num_real_levels < num_classes:
            
            ## Get quantizer
            self.quantizer = KBinsDiscretizer(n_bins=num_levels, encode='ordinal', strategy=strategy)
            self.quantizer.fit(filtered_traces.reshape(-1, 1))
            
            # Create loader
            loader = DataLoader(dataset, batch_size=32, shuffle=False)
            
            all_levels = []
            
            print("Getting patches")
            for batch in tqdm(loader):
                
                # get src and tgt
                src, tgt, _ = batch
                # cat src and tgt
                patches = torch.cat((src, tgt), dim=1)
                
                # get patches
                patches = patcher(patches)
                
                # get levels
                levels = self.get_levels(patches)
                all_levels.append(levels)
            
            # cat all levels
            all_levels = torch.cat(all_levels, dim=0)
            
            # get unique levels
            unique, counts = torch.unique(all_levels, return_counts=True)
            
            # exclude the levels that are used less than 10 times
            few_labels_mask = counts < 10
            
            # get the number of real levels
            num_real_levels = len(unique) - few_labels_mask.sum()
            
            # increase the number of levels
            num_levels += 1
        
        # get number of classes (used levels)
        assert num_real_levels == num_classes, f"The number of classes {num_classes} is not equal to the number of levels {num_real_levels}"
        self.num_classes = num_classes
        
        ## Assign levels to classes
        level2class = {}

        class_counts = {}
        class_id = -1
        # iterate over levels
        for level, level_count in zip(unique, counts):
            # if the level is used more than 10 times we use it as a class.
            # if not we skip or aggregate it to the previous class
            if level_count >= 10:
                class_id += 1
            
            if class_id in class_counts:
                class_counts[class_id if class_id > 0 else 0] += level_count.item()
            else:
                class_counts[class_id if class_id > 0 else 0] = level_count.item()
            
            print(f"Level {level} has {level_count} samples: we assign it to class {class_id if class_id > 0 else 0}")
            level2class[level.item()] = class_id if class_id > 0 else 0
            
        ## Assign classes to levels for reconstruction
        class2level = {}

        for level, class_ in level2class.items():
            if class_ in class2level:
                continue   
            class2level[class_] = level
            
        self.level2class = level2class
        self.class2level = class2level
        
        ## Get class weights
        self.class_counts = np.array(list(class_counts.values()))
        self.class_weights = self.class_counts.sum() / (self.num_classes*self.class_counts)
        
        # get the most used level
        self.most_frequent_class = self.class_counts.argmax().item()
        
    def get_most_frequent_class(self):
            return self.most_frequent_class
        
    def get_levels(self, patches, reduction='mean'):
    
        """
        Parameters
        -----------
        patches : np.ndarray
            is an array of shape [bs x num_patch x n_vars x patch_len]
        reduction : str
            is the reduction strategy to apply to the patch dimension
            
        Returns
        --------
        levels : np.ndarray
            is an array of shape [bs x num_patch x n_vars]
        """    
        
        patches = patches.cpu()
        
        # compute a single value for each patch
        if reduction == 'mean':
            patch_mean = patches.mean(dim=-1)
        elif reduction == 'max':
            patch_mean = patches.max(dim=-1)[0]
        elif reduction == 'min':
            patch_mean = patches.min(dim=-1)[0]
        elif reduction == 'median':
            patch_mean = patches.median(dim=-1)[0]
        else:
            raise ValueError(f"Reduction {reduction} not supported")
        patch_mean_shape = patch_mean.shape

        # get quantization levels
        levels = self.quantizer.transform(patch_mean.reshape(-1, 1))
        levels = torch.from_numpy(levels).long()
        levels = levels.reshape(patch_mean_shape)
        
        return levels
    
    def get_reconstruction(self, levels, patch_len):
    
        """
        Parameters
        -----------
        levels : np.ndarray
            is an array of shape [bs x num_patch x n_vars]
        patch_len : int
            is the length of the patch
        Returns
        --------
        rec : np.ndarray
            is an array of shape [bs x num_patch x n_vars x patch_len]
        """
            
        # get the reconstruction from the quantization levels
        patch_mean_shape = levels.shape
        rec = self.quantizer.inverse_transform(levels.reshape(-1, 1))
        rec = rec.reshape(patch_mean_shape)
        rec = torch.from_numpy(rec).float()
        rec = rec.unsqueeze(-1)
        rec = rec.expand(-1, -1, -1, patch_len)
        
        return rec
    
    def get_reconstruction_from_classes(self, classes, patch_len):
        
        # get levels from classes
        levels = torch.zeros_like(classes)
        for class_, level in self.class2level.items():
            levels[classes == class_] = level
        
        # get reconstruction
        rec = self.get_reconstruction(levels.numpy(), patch_len)
        
        return rec

    def get_labels(self, patches, mask):
        
        # compute a single value for each patch
        levels = self.get_levels(patches, reduction='mean')
        
        # convert levels to classes
        classes = torch.zeros_like(levels)
        for level, class_ in self.level2class.items():
            classes[levels == level] = class_
        
        # mask classes
        classes[torch.logical_not(mask)] = -100
        
        return levels, classes

class VectorQuantizer():
    
    def __init__(self, dataset, patcher, args_dict, vqvae_path, device):
        
        """
        Parameters
        -----------
        dataset : torch.utils.data.Dataset
            is the dataset to use to compute the quantization levels.
        patcher : Patcher
            is the patcher to use to get the patches
        args_dict : dict
            is the dictionary containing the arguments to use to create the VQVAE
        vqvae_path : str
            is the path to the VQVAE checkpoint
        device : str
            is the device to use. It can be 'cpu' or 'cuda'
        """
        
        # Load VQVAE Quantizer
        self.quantizer = Model(args_dict)   
        self.quantizer.to(device)
        self.quantizer.load_state_dict(torch.load(vqvae_path))
        
        # Create loader
        loader = DataLoader(dataset, batch_size=32, shuffle=False)
        
        # Get codes
        print("Getting patches")
        all_levels = []
        for batch in tqdm(loader):
            
            # Get batch (src, tgt, stim_id)
            src = batch['src']
            tgt = batch['tgt']
            stim_id = None
            
            # Move to device
            src = src.to(device)
            tgt = tgt.to(device)
            #stim_id = stim_id.to(device)
                    
            # Concat src and tgt
            patches = torch.cat((src, tgt), dim=-1)
            patches = patches.permute(0, 2, 1)
            
            # Get patches
            patches = patcher(patches)
            
            # Get levels
            levels = self.get_levels(patches, stim=None)
            all_levels.append(levels)
        
        # cat all levels
        all_levels = torch.cat(all_levels, dim=0)
        
        # get unique levels
        unique, counts = torch.unique(all_levels, return_counts=True)
        
        # get number of classes (used levels)
        self.num_classes = len(unique)
        self.class_counts = counts
        warnings.warn(f"The number of classes {self.num_classes} is not equal to the number of levels {args_dict['K']}")
        # get class weights
        self.class_weights = counts.sum() / (self.num_classes*counts)

        # We observe that only some levels are used
        # We can use a focal loss to penalize more the unused levels
        # But we need to exclude those levels that are never used
        self.level2class = {k: v for v, k in enumerate(unique.tolist())}
        self.class2level = {v: k for k, v in self.level2class.items()}
        
        # get the most used level
        self.most_frequent_class = counts.argmax().item()
        
    def get_most_frequent_class(self):
            return self.most_frequent_class
        
    def get_levels(self, patches, stim, return_embeddings = False):
    
        """
        Parameters
        -----------
        patches : np.ndarray
            is an array of shape [bs x num_patch x n_vars x patch_len]
            
        Returns
        --------
        levels : np.ndarray
            is an array of shape [bs x nvars x num_patch]
        """    
        
        if return_embeddings:
            with torch.no_grad():
                embeddings, codes = self.quantizer.encode(patches, stim=stim, return_embeddings=True)
                
            return embeddings, codes 
        else:
            with torch.no_grad():
                codes = self.quantizer.encode(patches, stim=stim)
            
            return codes
    
    def get_reconstruction(self, levels, stim):
    
        """
        Parameters
        -----------
        levels : np.ndarray
            is an array of shape [bs x nvars x num_patch]
        Returns
        --------
        rec : np.ndarray
            is an array of shape [bs x num_patch x n_vars x patch_len]
        """
            
        # get the reconstruction from the quantization levels
        with torch.no_grad():
            x_tilde = self.quantizer.decode(levels, stim=stim)
                
        return x_tilde
    
    def get_reconstruction_from_classes(self, src_levels, tgt_classes, stim):
    
        # get levels from classes
        tgt_levels = torch.zeros_like(tgt_classes)
        for class_, level in self.class2level.items():
            tgt_levels[tgt_classes == class_] = level
        
        # src levels 
        src_levels = src_levels.squeeze()
        levels = torch.cat((src_levels, tgt_levels), dim=2)
        
        # get reconstruction
        rec = self.get_reconstruction(levels, stim=stim)
        
        return rec

    def get_labels(self, patches, stim, mask):
        
        # compute a single value for each patch
        levels = self.get_levels(patches, stim=stim)
        # levels is of shape [bs x n_vars x num_patch]
        # permute
        mask = mask.permute(0, 2, 1)
        
        # convert levels to classes
        classes = torch.zeros_like(levels)
        for level, class_ in self.level2class.items():
            classes[levels == level] = class_
        
        # mask classes
        #labels = classes.clone()
        classes[torch.logical_not(mask)] = -100
        
        return levels, classes#, labels
    
    def get_embeddings(self, patches, stim):
        
        # compute a single value for each patch
        embeddings, levels = self.get_levels(patches, stim=stim, return_embeddings=True)
        # levels is of shape [bs x n_vars x num_patch]
        
        # convert levels to classes
        classes = torch.zeros_like(levels)
        for level, class_ in self.level2class.items():
            classes[levels == level] = class_
        
        return levels, classes, embeddings