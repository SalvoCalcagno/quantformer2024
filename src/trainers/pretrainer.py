import torch
import torch.nn as nn
import torch.optim as optim
import src.models as models

from tqdm.auto import tqdm
from src.saver import Saver
import torch.nn.functional as F
from src.datasets.allen_singlestimulus import STIM_RESPONSE_FRAMES
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.utils import init_best_metrics, update_best_metrics, compute_metrics, compute_classification_metrics, get_class_predictions, plot_timeseries

class Trainer:

    def __init__(self, args):
        
        # Store args
        self.args = args
        
        # Set criterion
        self.criterion = nn.MSELoss()
        
        # Create saver
        self.saver = Saver(args.logdir, args.exp_name)
        
        # Save args
        self.saver.save_configuration(vars(args))
        
        # Set Task
        self.cls = True
             
    def train(self, datasets):
        
        # Get splits
        splits = list(datasets.keys())
        
        # Get mode
        mode = "val" if "val" in splits else "test"
        
        if self.args.model in ['patchtst_pretrain', 'patchtst_finetune_nochans']:
        
            def collate_fn_eval(batch_l):
                
                # get number of neurons per sample
                num_neurons = [sample['src'].shape[0] for sample in batch_l]
                
                # expand stimulus
                stim = [torch.tensor(sample['stim']).expand(n, -1) for sample, n in zip(batch_l, num_neurons)]
                
                # init batch
                batch = {
                    'src': torch.vstack([torch.tensor(sample['src']) for sample in batch_l]),
                    'tgt': torch.vstack([torch.tensor(sample['tgt']) for sample in batch_l]),
                    'activation_labels': torch.cat([torch.tensor(sample['activation_labels']) for sample in batch_l]),
                    'stim': torch.vstack([torch.tensor(s) for s in stim]),
                }
                
                return batch
            
            def collate_fn(batch_l):
                
                #batch_l is the list of samples
                batch = {}
                num_neurons = [sample['src'].shape[0] for sample in batch_l]
                stim = [torch.tensor(sample['stim']).expand(n, -1) for sample, n in zip(batch_l, num_neurons)]
                
                batch['src'] = torch.vstack([torch.tensor(sample['src']) for sample in batch_l])
                batch['tgt'] = torch.vstack([torch.tensor(sample['tgt']) for sample in batch_l])
                batch['activation_labels'] = torch.cat([torch.tensor(sample['activation_labels']) for sample in batch_l])
                batch['stim'] = torch.vstack([torch.tensor(sample) for sample in stim])
                    
                # select all activations
                active_mask = batch['activation_labels'].bool()
                active_src = batch['src'][active_mask]
                active_tgt = batch['tgt'][active_mask]
                active_stim = batch['stim'][active_mask]
                active_labels = batch['activation_labels'][active_mask]
                
                # negative samples
                inactive_mask = ~active_mask
                inactive_src = batch['src'][inactive_mask]
                inactive_tgt = batch['tgt'][inactive_mask]
                inactive_stim = batch['stim'][inactive_mask]
                inactive_labels = batch['activation_labels'][inactive_mask]
                
                # shuffle negative samples
                idx = torch.randperm(inactive_src.shape[0])
                inactive_src = inactive_src[idx]
                inactive_tgt = inactive_tgt[idx]
                inactive_stim = inactive_stim[idx]
                inactive_labels = inactive_labels[idx]
                
                # select same number of negative samples
                n = active_src.shape[0]
                inactive_src = inactive_src[:n]
                inactive_tgt = inactive_tgt[:n]
                inactive_stim = inactive_stim[:n]
                inactive_labels = inactive_labels[:n]
                
                # merge active and inactive samples
                batch['src'] = torch.vstack([active_src, inactive_src])
                batch['tgt'] = torch.vstack([active_tgt, inactive_tgt])
                batch['stim'] = torch.vstack([active_stim, inactive_stim])
                batch['activation_labels'] = torch.cat([active_labels, inactive_labels])

                return batch
            
        elif self.args.model == 'patchtst_finetune':
            
            collate_fn = None
            collate_fn_eval = None

        
        # Create loaders
        loaders = {
            split: DataLoader(
                datasets[split],
                batch_size=self.args.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.args.workers,
                collate_fn=collate_fn if split=='train' else collate_fn_eval,
            ) for split in splits
        }
        # Save loaders
        self.loaders = loaders
        
        # Create model
        self.args.num_patch = 42
        module = getattr(models, self.args.model)
        model = getattr(module, "Model")(vars(self.args))
        
        # Move to device
        model = model.to(self.args.device)
        
        # If finetune, load pretrain weights
        if self.args.finetuning == 'linear':
            
            # Load pretrain model
            # Pretrained on 11 mice, drifting gratings, all data balanced. Task: event classification on tsai_wen labels
            model_path = "YOUR_PATH/patchtst_pretrain_best.pth"
            state_dict = torch.load(model_path, map_location=self.args.device)
            
            # Load weights
            model.load_state_dict(state_dict, strict=False)
            
            # Freeze model
            for param in model.model.parameters():
                param.requires_grad = False
            
        # Set optimizer
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        # Create optimizer
        optimizer_class = getattr(optim, self.args.optimizer)
        optimizer = optimizer_class([p for p in model.parameters() if p.requires_grad], **optim_params)

        # Configure scheduler
        if self.args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode = 'min', 
                patience=self.args.patience, 
                factor=self.args.reduce_lr_factor
            )
        else:
            scheduler = None
            

        # Initialize metrics
        metrics_to_minimize = []
        metrics_to_maximize = ['balanced_accuracy']
        
        best_metrics = init_best_metrics(
            metrics_to_minimize=metrics_to_minimize,
            metrics_to_maximize = metrics_to_maximize,
            mode=mode
        )
        
        # Get length of response
        self.response_len = STIM_RESPONSE_FRAMES[self.args.stimulus]
        
        # Watch model if enabled
        if self.args.watch_model:
            self.saver.watch_model(model)

        # Process each epoch
        try:
            for epoch in range(self.args.epochs):
                        
                # Epoch metrics
                self.epoch_metrics = {}
                # Epoch labels
                self.epoch_activation_labels = {}
                # Epoch predictions
                self.epoch_activation_preds = {}
                
                # Process each split
                for split in splits:

                    # Set network mode
                    if split == 'train':
                        model.train()
                        torch.set_grad_enabled(True)
                    else:
                        model.eval()
                        torch.set_grad_enabled(False)
                    
                    # Process each batch
                    for batch in tqdm(loaders[split]):
                        
                        # Forward Batch
                        batch_metrics, activation_preds, outputs = self.forward_batch(batch, model, optimizer, split)
                        
                        # Add metrics to epoch results
                        bs = batch['src'].shape[0]
                        for k, v in batch_metrics.items():
                            v *= bs
                            self.epoch_metrics[k] = self.epoch_metrics[k] + [v] if k in self.epoch_metrics else [v]
                        # Accumulate number of samples
                        self.epoch_metrics[f'{split}/num_samples'] = self.epoch_metrics[f'{split}/num_samples'] + bs if f'{split}/num_samples' in self.epoch_metrics else bs
                        

                        # Add activation predictions to epoch results
                        self.epoch_activation_labels[f'{split}/activation_labels'] = self.epoch_activation_labels[f'{split}/activation_labels'] + [batch['activation_labels']] if f'{split}/activation_labels' in self.epoch_activation_labels else [batch['activation_labels']]
                        self.epoch_activation_preds[f'{split}/activation_preds'] = self.epoch_activation_preds[f'{split}/activation_preds'] + [activation_preds] if f'{split}/activation_preds' in self.epoch_activation_preds else [activation_preds]
                    
                    # Plot train predictions
                    if split == 'train' and not self.cls:
                        fig = plot_timeseries(batch['src'], batch['tgt'], outputs, batch['activation_labels'], activation_preds)
                        self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                            
                    # Epoch end: compute epoch metrics
                    num_samples = self.epoch_metrics[f'{split}/num_samples']
                    del self.epoch_metrics[f'{split}/num_samples']
                    for k, v in self.epoch_metrics.items():
                        if k.startswith(split):
                            self.epoch_metrics[k] = sum(v)/num_samples
                    
                    # Stack activation labels and predictions
                    self.epoch_activation_labels[f'{split}/activation_labels'] = torch.cat(self.epoch_activation_labels[f'{split}/activation_labels'], dim=0)
                    self.epoch_activation_preds[f'{split}/activation_preds'] = torch.cat(self.epoch_activation_preds[f'{split}/activation_preds'], dim=0)
                    
                    # Compute classification metrics
                    self.epoch_metrics = {
                        **self.epoch_metrics, 
                        **compute_classification_metrics(
                            self.epoch_activation_labels[f'{split}/activation_labels'], 
                            self.epoch_activation_preds[f'{split}/activation_preds'], split
                            )
                        }
                                    
                # Add to saver
                for k, v in self.epoch_metrics.items():
                    self.saver.add_scalar(k, v, epoch)
                # Save learning rate
                self.saver.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                    
                best_metrics, save_flag = update_best_metrics(self.epoch_metrics, best_metrics, mode=mode)
                
                # Log best metrics
                for k, v in best_metrics.items():
                    self.saver.add_scalar(k, v, epoch)
                
                # Save model and plot if needed
                if save_flag:
                    self.saver.save_model(model, self.args.model, epoch)
                    # plot timeseries
                    if not self.cls:
                        fig = plot_timeseries(batch['src'], batch['tgt'], outputs, batch['activation_labels'], activation_preds)
                        self.saver.add_plot(f'{mode}/sample_plots', fig, epoch) 


                # log all metrics
                self.saver.log()            

                # Check LR scheduler
                if scheduler is not None:
                    scheduler.step()
        
        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass

        except FloatingPointError as err:
            print(f'Error: {err}')
            
        return model, None
    
    def forward_batch(self, batch, model, optimizer, split):
            
            if self.args.model == 'patchtst_pretrain':
                return self.forward_batch_pretrain(batch, model, optimizer, split)
            elif self.args.model == 'patchtst_finetune':
                return self.forward_batch_finetune(batch, model, optimizer, split)
            elif self.args.model == 'patchtst_finetune_nochans':
                return self.forward_batch_finetune_nochans(batch, model, optimizer, split)
    
    def forward_batch_pretrain(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        tgt = batch['tgt'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
                    
        # Forward
        outputs = model(tgt).squeeze()
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, activation_labels.float())
        
        if split == 'train':
        
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        batch_metrics = {
            f"{split}/loss": loss.item(),
        }
        
        # Get activation predictions
        activation_preds = (outputs>0).long() 
        
        return batch_metrics, activation_preds, outputs
      
    def forward_batch_finetune(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        
        outputs = model(src, stim)
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, activation_labels.float())
        
        # Optimize
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        batch_metrics = {
            f"{split}/loss": loss.item(),
        }
        
        # Get activation predictions
        activation_preds = (outputs>0).long() 
        
        return batch_metrics, activation_preds, outputs
    
    def forward_batch_finetune_nochans(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
                    
        # Forward
        outputs = model(src, stim).squeeze()
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, activation_labels.float())
        
        if split == 'train':
        
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        batch_metrics = {
            f"{split}/loss": loss.item(),
        }
        
        # Get activation predictions
        activation_preds = (outputs>0).long() 
        
        return batch_metrics, activation_preds, outputs