import json

import torch
import torch.nn as nn
import torch.optim as optim
import src.models as models

from tqdm.auto import tqdm
from src.saver import Saver
import torch.nn.functional as F
from src.datasets.allen_singlestimulus import STIM_RESPONSE_FRAMES, BASELINE_FRAMES
from torch.utils.data import DataLoader, WeightedRandomSampler
from src.models.brainlm_mae.modeling_brainlm import BrainLMForPretraining, BrainLMForResponseClassification
from transformers import PretrainedConfig
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
        
        self.cls = args.cls
        
    def train(self, datasets):
        
        # Get splits
        splits = list(datasets.keys())
        
        # Get mode
        mode = "val" if "val" in splits else "test"
        
        # Create loaders
        loaders = {
            split: DataLoader(
                datasets[split],
                batch_size=self.args.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.args.workers
            ) for split in splits
        }
        # Save loaders
        self.loaders = loaders
        
        model_class = BrainLMForResponseClassification if self.args.cls else BrainLMForPretraining
        
        # Open config
        with open("brainlm/checkpoint/config.json", "r") as f:
            config = json.load(f)
        # Change stim size
        config["stim_dim"] = self.args.stim_size
        # Save
        with open("brainlm/checkpoint/config.json", "w") as f:
            json.dump(config, f)
        
        # Create model
        if self.args.finetuning in ["linear", "e2e"]:
            model = model_class.from_pretrained('brainlm/checkpoint')
        else:
            # random init
            config = PretrainedConfig.from_json_file('brainlm/checkpoint/config.json')
            model = model_class(config)
        # Adapt model to data
        batch = next(iter(loaders['train']))
        xyz_vectors = batch['xyz_vectors']
        _, num_brain_voxels, _ = xyz_vectors.shape
        model.num_brain_voxels = num_brain_voxels
        if not self.args.cls:
            model.decoder.num_brain_voxels = num_brain_voxels
            model.decoder.num_masked_pacthes = num_brain_voxels * (model.config.num_last_timepoints_masked // model.config.timepoint_patching_size)
        
        # Move to device
        model = model.to(self.args.device)
        
        # Set optimizer
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}
            
        if self.args.finetuning == 'linear':
            
            # freeze the model
            for param in model.parameters():
                param.requires_grad = False
            
            # unfreeze the last layer
            """
            for param in model.decoder.decoder_pred1.parameters():
                param.requires_grad = True
            for param in model.decoder.decoder_pred_nonlinearity.parameters():
                param.requires_grad = True
            for param in model.decoder.decoder_pred2.parameters():
                param.requires_grad = True
            """
            for param in model.decoder.stim_embedding.parameters():
                param.requires_grad = True
            for param in model.decoder.linear_head_with_stim.parameters():
                param.requires_grad = True
            #for param in model.decoder.decoder_pred_nonlinearity2.parameters():
            #    param.requires_grad = True
            
            
            
            

        # Create optimizer
        optimizer_class = getattr(optim, self.args.optimizer)
        otimizable_params = [p for p in model.parameters() if p.requires_grad]
        optimizer = optimizer_class(otimizable_params, **optim_params)

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
        if self.cls:
            metrics_to_minimize = []
            metrics_to_maximize = ['accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
        else:
            metrics_to_minimize = ['mse', 'mae', 'mase_f0', 'mase_avg', 'smape', 'dtw']
            metrics_to_maximize = ['correlation', 'ssim', 'psnr', 'accuracy', 'precision', 'recall', 'f1', 'balanced_accuracy']
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
                    if not self.args.cls:
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
        if self.args.cls:
            return self.forward_batch_classification(batch, model, optimizer, split)
        else:
            return self.forward_batch_forecasting(batch, model, optimizer, split)
        
    def forward_batch_classification(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        #tgt = batch['tgt'].to(self.args.device) 
        #tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        xyz_vectors = batch['xyz_vectors'].to(self.args.device)/512
        
        # Prepare BrainLM input
        #x = torch.cat([src, tgt], dim=-1)
         
        loss, outputs = model(src, xyz_vectors, output_hidden_states=True, stim=stim, labels=activation_labels)        
           
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
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
        activation_preds = (outputs.argmax(-1)).long()
                        
        return batch_metrics, activation_preds, outputs
        
    def forward_batch_forecasting(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        tgt = batch['tgt'].to(self.args.device)
        tgt_avg = src.mean(dim=-1).unsqueeze(-1).repeat(1, 1, self.args.forecast_window).to(self.args.device)
        #tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        #activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        xyz_vectors = batch['xyz_vectors'].to(self.args.device)/512
        
        # Prepare BrainLM input
        x = torch.cat([src, tgt], dim=-1)
         
        outputs = model(x, xyz_vectors, output_hidden_states=True, stim=stim)        
        
        # Compute loss
        loss = outputs.loss
        outputs, _ = outputs.logits
        outputs = outputs.reshape(src.shape[0], src.shape[1], -1)
           
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Optimize
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        if self.args.stimulus != "locally_sparse_noise":
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg, split),
                #**compute_metrics(src, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split, suffix='_avg')
            }
        else:
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg, split),
            }
            
        # Get activation predictions
        activation_preds = get_class_predictions(outputs.cpu(), self.loaders[split].dataset.activation_threshold, eval_len=self.response_len)
                        
        return batch_metrics, activation_preds, outputs
    
    def forward_batch_autoformer(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src']#.to(self.args.device)
        tgt = batch['tgt']#.to(self.args.device)
        tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        
        # prepare informer input
        time = torch.arange(0, self.args.window_size/30, 1/30)
        context_len = self.args.source_length//2
        
        x = src.transpose(1, 2)
        x_mark = time[:self.args.source_length].unsqueeze(0).unsqueeze(-1)
        x_mark = x_mark.expand(x.size(0), -1, x.size(-1))
        
        y = tgt.transpose(1, 2)
        context = x[:,-context_len:]
        y = torch.cat((context, y), dim=1)
        y_mark = time[context_len:].unsqueeze(0).unsqueeze(-1)
        y_mark = y_mark.expand(y.size(0), -1, y.size(-1))
        
        # move to device
        x = x.to(self.args.device)
        x_mark = x_mark.to(self.args.device)
        y = y.to(self.args.device)
        y_mark = y_mark.to(self.args.device)
        tgt = tgt.to(self.args.device)
         
        outputs = model(x, x_mark, y, y_mark, stim=stim).transpose(1, 2)
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, tgt)
        
        # Optimize
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        if self.args.stimulus != "locally_sparse_noise":
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
                **compute_metrics(src, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split, suffix='_avg')
            }
        else:
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
            }
            
        # Get activation predictions
        activation_preds = get_class_predictions(outputs.cpu(), self.loaders[split].dataset.activation_threshold, eval_len=self.response_len)
                        
        return batch_metrics, activation_preds, outputs
        
    def forward_batch_crossformer(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        tgt = batch['tgt'].to(self.args.device)
        tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        
        outputs = model(src, stim)
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, tgt)
        
        # Optimize
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        if self.args.stimulus != "locally_sparse_noise":
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
                **compute_metrics(src, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split, suffix='_avg')
            }
        else:
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
            }
            
        # Get activation predictions
        activation_preds = get_class_predictions(outputs.cpu(), self.loaders[split].dataset.activation_threshold, eval_len=self.response_len)
                        
        return batch_metrics, activation_preds, outputs
    
    def forward_batch_patchtst(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        tgt = batch['tgt'].to(self.args.device)
        tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        
        outputs = model(src, stim)
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, tgt)
        
        # Optimize
        if split == 'train':
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        # Compute metrics
        if self.args.stimulus != "locally_sparse_noise":
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
                **compute_metrics(src, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split, suffix='_avg')
            }
        else:
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
            }
            
        # Get activation predictions
        activation_preds = get_class_predictions(outputs.cpu(), self.loaders[split].dataset.activation_threshold, eval_len=self.response_len)
                        
        return batch_metrics, activation_preds, outputs
    
    def forward_batch_lstm_classification(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        tgt = batch['tgt'].to(self.args.device)
        tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
            
        # Forward
        outputs = model(src, stim)
        
        # Check NaN
        if torch.isnan(outputs).any():
            raise FloatingPointError('Found NaN values')
        
        # Compute loss
        loss = model.loss(outputs, activation_labels.long())
        
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
        activation_preds = (outputs.argmax(-1)).long() 
        
        return batch_metrics, activation_preds, outputs
         
    def forward_batch_lstm(self, batch, model, optimizer, split):
        
        # Unpack inputs and move to device
        src = batch['src'].to(self.args.device)
        tgt = batch['tgt'].to(self.args.device)
        tgt_avg = batch['mean_trace'].to(self.args.device)
        activation_labels = batch['activation_labels'].to(self.args.device)
        activation_probs = batch['activation_probs'].to(self.args.device)
        stim = batch['stim'].to(self.args.device)
        
        if split == 'train': # use teacher forcing
            
            # Forward
            outputs = model(src, tgt, stim) #TODO: use real model
            
            # Check NaN
            if torch.isnan(outputs).any():
                raise FloatingPointError('Found NaN values')
            
            # Compute loss
            loss = model.loss(outputs, tgt)
        
            # Optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        
        else: # use auto-regressive prediction
            
            # Initialize outputs
            outputs = torch.zeros_like(tgt)
            tgt_tmp  = src[:, :, -1:]
            
            # Forward
            for t in range(self.args.forecast_window):
                
                # Forward
                output = model(src[:, :, :-1], tgt_tmp, stim)
                
                # Check NaN
                if torch.isnan(output).any():
                    raise FloatingPointError('Found NaN values')
                
                # Store output
                tgt_tmp = torch.cat([tgt_tmp, output[:, :, -1:]], dim=-1)
                outputs[:, :, t] = output[:, :, -1]
                
                # Compute loss
                loss = model.loss(outputs, tgt) 
        
        # Compute metrics
        if self.args.stimulus != "locally_sparse_noise":
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
                **compute_metrics(src, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split, suffix='_avg')
            }
        else:
            batch_metrics = {
                f"{split}/loss": loss.item(),
                **compute_metrics(src, tgt, outputs, tgt_avg[:, :, BASELINE_FRAMES : BASELINE_FRAMES + self.args.forecast_window], split),
            }
            
        # Get activation predictions
        activation_preds = get_class_predictions(outputs.cpu(), self.loaders[split].dataset.activation_threshold, eval_len=self.response_len)
        
        return batch_metrics, activation_preds, outputs