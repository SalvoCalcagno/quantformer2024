import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import src.models as models
from src.saver import Saver
from sklearn import metrics
from pathlib import Path
from torch.utils.data import DataLoader
from src.datasets.quantizers import Quantizer, VectorQuantizer
from neuralforecast.losses.pytorch import *
from src.metrics.FocalLoss import FocalLoss
from sklearn.preprocessing import KBinsDiscretizer
from src.models.patchtst_self import Patch, PatchMask
from src.metrics.PretrainLoss import PretrainLoss, MaskedMSELoss
from src.metrics.correlation import TimeSeriesPearsonCorrCoef
from src.utils import plot_timeseries, plot_timeseries_causal_mask_quantized#_quantized
from src.utils import get_class_predictions, compute_classification_metrics
from src.datasets.allen_singlestimulus import STIM_RESPONSE_FRAMES, BASELINE_FRAMES


from src.metrics.MASE import MASE, SMAPE
from src.metrics.PSNR import compute_psnr_4d, compute_psnr_3d
from src.metrics.SSIM import compute_ssim_4d, compute_ssim_3d

class Trainer():
    
    def __init__(self, args):
        
        # Store Args
        self.args = args
        
        # Set criterion
        self.criterion = self.args.criterion
        
        # Create saver
        self.saver = Saver(args.logdir, args.exp_name)
        # Save args
        self.saver.save_configuration(vars(args))
        
    def train(self, datasets):
        
        if self.args.cls:
            return self.train_qclassifier(datasets)
        else:
            return self.train_quantcoder(datasets)
       
    def train_qclassifier(self, datasets):
        
        # Get splits
        splits = list(datasets.keys())

        if self.args.no_valid:
            val_name = 'val' if 'val' in splits else 'valid'
        
        # Create loaders
        loaders = {
            split: DataLoader(
                datasets[split],
                batch_size=self.args.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.args.workers
            ) for split in splits
        }
        
        patch = Patch(
            self.args.window_size,
            self.args.patch_len,
            self.args.patch_len,
        )
        
        if not self.args.random_init:
            quantizer_path = Path(self.args.quantizer)/ "quantformer_best.pth" 
            
        if not self.args.ablate_quantizer:
            
            # Get quantizer
            quantizer = VectorQuantizer(
                dataset = datasets['train'],
                patcher = patch,
                args_dict=vars(self.args),
                vqvae_path=quantizer_path,
                device=self.args.device
            )
            
            # Get the most frequent class
            most_frequent_class = quantizer.get_most_frequent_class()

            # Create model
            print("Your quantizer is using the following number of classes on this dataset: ", quantizer.num_classes)
        print("Your model wants to predict the following number of classes: ", self.args.num_classes)
       
        module = getattr(models, self.args.model)
        model = getattr(module, "Model")(vars(self.args))
        
        # Move to device
        model = model.to(self.args.device)

        # Set criterion (focal loss is default)    
        self.criterion = nn.CrossEntropyLoss()
       
        if not self.args.random_init:
            # Use pretrained weights
            state_dict = torch.load(quantizer_path)
            # load params from state dict to model if the key is in the model
            if self.args.ablate_quantizer:
                model.load_state_dict({
                    k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith('stimulus_embedding') and not k.startswith('codebook')},
                    strict=False
                )
            else:
                model.load_state_dict({
                    k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith('stimulus_embedding')},
                    strict=False
                )
    
        # Set optimizer
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        # Create optimizer
        optimizer_class = getattr(optim, self.args.optimizer)
        optimizer = optimizer_class(model.parameters(), **optim_params)

        # Configure scheduler
        if self.args.warmup_epochs > 0:
            print(f"warmup for {self.args.warmup_epochs} epochs")
            scheduler = optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.args.warmup_factor,
                total_iters=self.args.warmup_epochs
            )
        # Configure scheduler
        elif self.args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode = 'min', 
                patience=self.args.patience, 
                factor=self.args.reduce_lr_factor
            )
        else:
            scheduler = None

        # Initialize the final result metrics
        result_metrics = { split: {} for split in splits }
        
        if self.args.no_valid:
            # Test metrics
            max_test_acc = 0.0
            max_test_balanced_acc = 0.0
        else:
            # Validation Metrics
            max_valid_acc = 0.0

            # Test Metrics
            test_acc_at_max_valid_acc = 0.0

        # Train Metrics
        lowest_train_loss = float('inf')
    
        # Train the model
        try:
            for epoch in range(self.args.epochs):

                for split in splits:
                    
                    # Initialize epoch metrics (split-specific)
                    epoch_metrics = {}
                    epoch_activation_preds = []
                    epoch_activation_labels = []

                    # Set model mode
                    if split == 'train':
                        model.train()
                    elif not self.args.disable_eval_call:
                        model.eval()

                    # Iterate over batches
                    for batch in tqdm(loaders[split]):

                        # Get data from batch
                        src = batch['src']
                        #tgt = batch['tgt']
                        stim = batch['stim']
                        activation_labels = batch['activation_labels']
                        src = src.permute(0, 2, 1)
                        src_patch = patch(src)

                        # Move to device
                        src_patch = src_patch.to(self.args.device)
                        stim = stim.to(self.args.device).float()
                        activation_labels = activation_labels.to(self.args.device)
                        
                        # Reset gradients
                        optimizer.zero_grad()

                        # Forward Pass
                        _, logits = model(src_patch, stimulus=stim)
                        activation_preds = logits.argmax(dim=-1)
                                                
                        # Get dimensions
                        #bs, _, nvars, _ = src_patch.shape
                         
                        # Compute loss
                        loss = self.criterion(logits.transpose(1, -1), activation_labels)

                        # Backward pass
                        if split == 'train':
                            loss.backward()
                            optimizer.step()

                        # Compute batch metrics
                        batch_metrics = {
                            'loss': loss.item()
                        }

                        epoch_activation_preds.append(activation_preds)
                        epoch_activation_labels.append(activation_labels)
                        
                        # Update epoch metrics
                        for k, v in batch_metrics.items():
                            v = v * src.shape[0]
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                    # Compute Epoch Metrics
                    num_samples = len(loaders[split].dataset) if not loaders[split].drop_last else len(loaders[split]) *self.args.batch_size
                    for k, v in epoch_metrics.items():
                        epoch_metrics[k] = sum(v) / num_samples
                        # Add to Saver
                        self.saver.add_scalar(f'{split}/{k}', epoch_metrics[k], epoch)
                        
                    # Stack activation labels and preds
                    epoch_activation_labels = torch.cat(epoch_activation_labels, dim=0)
                    epoch_activation_preds = torch.cat(epoch_activation_preds, dim=0)
                    
                    # Compute activation metrics
                    activation_metrics = compute_classification_metrics(
                        epoch_activation_labels, 
                        epoch_activation_preds,
                        split,
                        split_in_name=False
                    )
                    
                    epoch_metrics = {**epoch_metrics, **activation_metrics}
                    
                    # Add to Saver
                    for k, v in activation_metrics.items():
                        self.saver.add_scalar(f'{split}/{k}', v, epoch)
                    
                    # Update result metrics
                    for metric in epoch_metrics:
                        if metric not in result_metrics[split]:
                            result_metrics[split][metric] = [epoch_metrics[metric]]
                        else:
                            result_metrics[split][metric].append(epoch_metrics[metric])
                                    
                # Add learning rate to Saver
                self.saver.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                if self.args.no_valid:
                    
                    if epoch >= self.args.warmup_epochs:
                        
                        saved = False
                        
                        # Update best metrics
                        if result_metrics['test']['balanced_accuracy'][-1] > max_test_balanced_acc:
                            max_test_balanced_acc = result_metrics['test']['balanced_accuracy'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/max_balanced_accuracy', max_test_balanced_acc, epoch)
                
                else:
                    if epoch >= self.args.warmup_epochs:
                        
                        saved = False
                        if result_metrics[val_name]['smape'][-1] < lowest_valid_smape:
                            lowest_valid_smape = result_metrics[val_name]['smape'][-1]
                            test_smape_at_lowest_valid_smape = result_metrics['test']['smape'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/smape_at_lowest_valid_smape', test_smape_at_lowest_valid_smape, epoch)
                        self.saver.add_scalar('valid/lowest_valid_smape', lowest_valid_smape, epoch)

                if result_metrics['train']['loss'][-1] < lowest_train_loss:
                    lowest_train_loss = result_metrics['train']['loss'][-1]
                    #save model
                    self.saver.save_model(model, self.args.model, epoch, model_name=f"{self.args.model}_pretrained")

                # Add to Saver
                self.saver.add_scalar('train/lowest_train_loss', lowest_train_loss, epoch)

                # Log all metrics
                self.saver.log()
                # Update scheduler
                if self.args.warmup_epochs > 0:
                    scheduler.step()
                elif self.args.use_scheduler:
                    #TODO: Check if it is the right metric: it is not
                    # We don't use scheduler for now
                    scheduler.step(lowest_test_mse if self.args.no_valid else lowest_valid_mse)
                
                # Save model
                if epoch % self.args.save_every == 0:
                    self.saver.save_model(model, self.args.model, epoch, model_name=f"{self.args.model}_pretrained")

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass
        
        except FloatingPointError as err:
            print(f"Error: {err}")

        if self.args.no_valid:
            # print main metrics
            print(f"Highest Test Accuracy: {max_test_acc}")
            print(f"Highest Test Balanced Accuracy: {max_test_balanced_acc}")
        else:
            # print main metrics
            pass
        
        return model, result_metrics
        
    def train_quantcoder(self, datasets):
        
        # Get splits
        splits = list(datasets.keys())

        if self.args.no_valid:
            val_name = 'val' if 'val' in splits else 'valid'
        
        # Create loaders
        loaders = {
            split: DataLoader(
                datasets[split],
                batch_size=self.args.batch_size,
                shuffle=(split == 'train'),
                num_workers=self.args.workers
            ) for split in splits
        }
        
        # Compute mask ratio as the ratio of forecast window to window size
        mask_ratio = self.args.forecast_window / self.args.window_size
        # Create patching operations
        masked_patch = PatchMask(
            self.args.patch_len,
            self.args.patch_len,
            mask_ratio,
            force_causal_masking=True,
            mask_on_peaks=self.args.mask_on_peaks
        )
        
        patch = Patch(
            self.args.window_size,
            self.args.patch_len,
            self.args.patch_len,
        )
        
        if not self.args.random_init:
            quantizer_path = Path(self.args.quantizer)/ "quantformer_best.pth" 
            
        if not self.args.ablate_quantizer:
            # Get quantizer
            quantizer = VectorQuantizer(
                dataset = datasets['train'],
                patcher = patch,
                args_dict=vars(self.args),
                vqvae_path=quantizer_path,
                device=self.args.device
            )
            
            # Get the most frequent class
            most_frequent_class = quantizer.get_most_frequent_class()

            # Create model
            self.args.num_classes = quantizer.num_classes
            
            print("Your quantizer is using the following number of classes on this dataset: ", self.args.num_classes)
        
        # Response len (on which evaluate activation)
        self.response_len = STIM_RESPONSE_FRAMES[self.args.stimulus]
        
        module = getattr(models, self.args.model)
        model = getattr(module, "Model")(vars(self.args))
        
        # Move to device
        model = model.to(self.args.device)

        # Set criterion (focal loss is default)
        if self.args.criterion == 'mse':
            self.criterion = nn.MSELoss()
            
        elif self.args.criterion == 'focal':
            alpha = quantizer.class_weights
            alpha = alpha.to(self.args.device)
            gamma = 0.0
            self.criterion = FocalLoss(
                alpha=alpha, 
                gamma=gamma, 
                reduction='mean'
            )
            #self.criterion = nn.CrossEntropyLoss()
            
        elif self.args.criterion == 'cross_entropy':
            self.criterion = nn.CrossEntropyLoss()
        
        else:
            raise NotImplementedError(f"{self.args.criterion} is not implemented")
                
        
        # Compute number of predicted patches
        num_predicted_patches = self.args.forecast_window // self.args.patch_len
        
        if not self.args.random_init:
            state_dict = torch.load(quantizer_path)
            if self.args.ablate_quantizer:
                model.load_state_dict({
                    k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith('stimulus_embedding') and not k.startswith('codebook')},
                    strict=False
                )
            else:
                model.load_state_dict({
                    k: v for k, v in state_dict.items() if k in model.state_dict() and not k.startswith('stimulus_embedding')},
                    strict=False
                )
        
        #for param in model.parameters():
        #    if not param.requires_grad:
        #        print(param.shape)
                
        # Set optimizer
        optim_params = {'lr': self.args.lr, 'weight_decay': self.args.weight_decay}
        if self.args.optimizer == 'Adam':
            optim_params = {**optim_params, 'betas': (0.9, 0.999)}
        elif self.args.optimizer == 'SGD':
            optim_params = {**optim_params, 'momentum': 0.9}

        # Create optimizer
        optimizer_class = getattr(optim, self.args.optimizer)
        optimizer = optimizer_class(model.parameters(), **optim_params)

        # Configure scheduler
        if self.args.warmup_epochs > 0:
            print(f"warmup for {self.args.warmup_epochs} epochs")
            scheduler = optim.lr_scheduler.ConstantLR(
                optimizer,
                factor=self.args.warmup_factor,
                total_iters=self.args.warmup_epochs
            )
        # Configure scheduler
        elif self.args.use_scheduler:
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                optimizer, 
                mode = 'min', 
                patience=self.args.patience, 
                factor=self.args.reduce_lr_factor
            )
        else:
            scheduler = None

        # Initialize the final result metrics
        result_metrics = { split: {} for split in splits }
        
        compute_mase = MASE()
        compute_smape = SMAPE()
        compute_correlation = TimeSeriesPearsonCorrCoef()

        if self.args.no_valid:
            # Test metrics
            lowest_test_mae = float('inf')
            lowest_test_mse = float('inf')
            lowest_test_mase = float('inf')
            lowest_test_smape = float('inf')
            max_test_acc = 0.0
            max_test_balanced_acc = 0.0
        else:
            # Validation Metrics
            lowest_valid_mae = float('inf')
            lowest_valid_mse = float('inf')
            lowest_valid_mase = float('inf')
            lowest_valid_smape = float('inf')
            max_valid_acc = 0.0

            # Test Metrics
            test_mae_at_lowest_valid_mae = float('inf')
            test_mse_at_lowest_valid_mse = float('inf')
            test_mase_at_lowest_valid_mase = float('inf')
            test_smape_at_lowest_valid_smape = float('inf')
            test_acc_at_max_valid_acc = 0.0

        # Train Metrics
        lowest_train_loss = float('inf')
    
        # Train the model
        try:
            for epoch in range(self.args.epochs):
                
                # empty cache
                #torch.cuda.empty_cache()
                for split in splits:
                    
                    # Initialize epoch metrics (split-specific)
                    epoch_metrics = {}
                    epoch_preds = []
                    epoch_labels = []
                    epoch_activation_preds = []
                    epoch_activation_labels = []

                    # Set model mode
                    if split == 'train':
                        model.train()
                    elif not self.args.disable_eval_call:
                        model.eval()

                    # Iterate over batches
                    for batch in tqdm(loaders[split]):

                        # Get data from batch
                        src = batch['src']
                        tgt = batch['tgt']
                        stim = batch['stim']
                        activation_labels = batch['activation_labels']
                        src = torch.cat((src, tgt), dim=-1)
                        src = src.permute(0, 2, 1)
                        src_patch, src_mask, mask = masked_patch(src)

                        # Move to device
                        src_patch = src_patch.to(self.args.device)
                        src_mask = src_mask.to(self.args.device)
                        mask = mask.to(self.args.device)
                        stim = stim.to(self.args.device).float()
                        
                        # Reset gradients
                        optimizer.zero_grad()
                        
                        if not self.args.ablate_quantizer:
                            # Get quantization levels and labels (they may be the same)
                            levels, labels = quantizer.get_labels(patches=src_patch, stim=None, mask=mask)
                            labels = labels.squeeze(-1)
                            #labels = labels.to(self.args.device)
                            #levels = levels.to(self.args.device)
                            
                            # Get Reconstruction from levels
                            rec_true = quantizer.get_reconstruction(levels, stim=None)

                            # Forward Pass
                            levels, logits = model(src_mask, stimulus=stim, mask = mask)
                            preds = logits.argmax(dim=-1)
                            
                            tgt_classes = preds[:, :, -num_predicted_patches:]
                            src_levels = levels[:, :, :-num_predicted_patches]
                                                    
                            # Get Reconstruction from Predicted Classes
                            rec_pred = quantizer.get_reconstruction_from_classes(src_levels, tgt_classes, stim=None)
                        
                        else:
                            
                            _, rec_pred = model(src_mask, stimulus=stim, mask = mask)
                            
                        # Get dimensions
                        bs, _, nvars, _ = src_patch.shape
                         
                        # Compute loss
                        if self.args.criterion == 'mse':
                            #loss = self.criterion(rec_pred, src_patch)
                            loss = MaskedMSELoss()(rec_pred, src_patch, mask)
                        else:
                            logits = logits.permute(0, 3, 1, 2)
                            
                            """
                            bs, _, nvars, _ = rec_pred.shape
 
                            loss = nn.CosineEmbeddingLoss()(
                                rec_pred.permute(0, 2, 1, 3).reshape(bs*nvars, -1),
                                src_patch.permute(0, 2, 1, 3).reshape(bs*nvars, -1),
                                torch.ones(bs*nvars).to(self.args.device)
                            )
                            """
                     
                            loss = self.criterion(logits, labels)
                            
                            #loss = loss + cosine_loss
                        
                        
                        # Backward pass
                        if split == 'train':
                            loss.backward()
                            optimizer.step()
                        
                        rec_pred = rec_pred.detach()  
                        # Get activation predictions (based on reconstruction)
                        activation_preds = get_class_predictions(
                            rec_pred.permute(0, 2, 1, 3).reshape(bs, nvars, -1),
                            loaders[split].dataset.activation_threshold,
                            eval_len = self.response_len
                        )

                        # Compute batch metrics
                        mse = ((rec_pred - src_patch)**2)[mask].mean()
                        mae = (rec_pred - src_patch).abs()[mask].mean()
                        if not self.args.ablate_quantizer:
                            mse_q = ((rec_pred - rec_true)**2)[mask].mean()
                            mae_q = (rec_pred - rec_true).abs()[mask].mean()
                       
                        rmse = mse.sqrt()
                        mase = compute_mase(rec_pred, src_patch)
                        smape = compute_smape(rec_pred, src_patch) 
                        
                        if not self.args.ablate_quantizer:
                            rmse_q = mse_q.sqrt()
                            mase_q = compute_mase(rec_pred, rec_true)
                            smape_q = compute_smape(rec_pred, rec_true)
                        
                        #rec_pred is of shape [bs x num_patch x n_vars x patch_len]
                        correlation = compute_correlation(rec_pred[:,  -num_predicted_patches:, :, :], src_patch[:,  -num_predicted_patches:, :, :], patched=False)
                        patch_correlation = compute_correlation(rec_pred[:,  -num_predicted_patches:, :, :], src_patch[:,  -num_predicted_patches:, :, :], patched=True)
                        psnr = compute_psnr_4d(rec_pred[:,  -num_predicted_patches:, :, :], src_patch[:,  -num_predicted_patches:, :, :])
                        ssim = compute_ssim_4d(rec_pred[:,  -num_predicted_patches:, :, :], src_patch[:,  -num_predicted_patches:, :, :], value_range=1.0)#value_range=src_patch.max() - src_patch.min())
                        
                        if not self.args.ablate_quantizer:
                            batch_metrics = {
                                'loss': loss.item(),
                                'mse': mse.item(),
                                'rmse': rmse.item(),
                                'mae': mae.item(),
                                'mase': mase.item(),
                                'smape': smape.item(),
                                'mse_q': mse_q.item(),
                                'rmse_q': rmse_q.item(),
                                'mae_q': mae_q.item(),
                                'mase_q': mase_q.item(),
                                'smape_q': smape_q.item(),
                                'correlation': correlation.item(),
                                'patch_correlation': patch_correlation.item(),
                                'psnr': psnr.item(),
                                'ssim': ssim.item()
                            }
                        else:
                            batch_metrics = {
                                'loss': loss.item(),
                                'mse': mse.item(),
                                'rmse': rmse.item(),
                                'mae': mae.item(),
                                'mase': mase.item(),
                                'smape': smape.item(),
                                'correlation': correlation.item(),
                                'patch_correlation': patch_correlation.item(),
                                'psnr': psnr.item(),
                                'ssim': ssim.item()
                            }

                        if not self.args.ablate_quantizer:
                            l_mask = labels != -100
                            epoch_preds.append(preds[l_mask].cpu().detach().numpy()) #N, C, k1, k2...
                            epoch_labels.append(labels[l_mask].cpu().detach().numpy()) #N, k1, k2...
                        epoch_activation_preds.append(activation_preds)
                        epoch_activation_labels.append(activation_labels)
                        
                        # Update epoch metrics
                        for k, v in batch_metrics.items():
                            v = v * src.shape[0]
                            epoch_metrics[k] = epoch_metrics[k] + [v] if k in epoch_metrics else [v]

                    # Compute Epoch Metrics
                    num_samples = len(loaders[split].dataset) if not loaders[split].drop_last else len(loaders[split]) *self.args.batch_size
                    for k, v in epoch_metrics.items():
                        epoch_metrics[k] = sum(v) / num_samples
                        # Add to Saver
                        self.saver.add_scalar(f'{split}/{k}', epoch_metrics[k], epoch)

                    if not self.args.ablate_quantizer:

                        # Aggregate logits and labels
                        epoch_labels = np.concatenate(epoch_labels) #N, k1, k2...
                        epoch_preds = np.concatenate(epoch_preds) #N, k1, k2...

                        # Flatten labels and preds
                        epoch_labels = epoch_labels.reshape(-1) #N*k1*k2...
                        epoch_preds = epoch_preds.reshape(-1) #N*k1*k2...

                        # Accuracy
                        accuracy = metrics.accuracy_score(epoch_labels, epoch_preds)
                        epoch_metrics['accuracy'] = accuracy
                        self.saver.add_scalar(f"{split}/accuracy", accuracy, epoch)

                        # Balanced Accuracy
                        balanced_accuracy = metrics.balanced_accuracy_score(epoch_labels, epoch_preds)
                        epoch_metrics['balanced_accuracy'] = balanced_accuracy
                        self.saver.add_scalar(f"{split}/balanced_accuracy", balanced_accuracy, epoch)
                        
                        # Confusion Matrix
                        #self.saver.add_confusion_matrix(epoch_labels, epoch_preds, classes=np.arange(quantizer.num_classes).tolist(), split=split, iter_n=epoch)
                    
                    # Stack activation labels and preds
                    epoch_activation_labels = torch.cat(epoch_activation_labels, dim=0)
                    epoch_activation_preds = torch.cat(epoch_activation_preds, dim=0)
                    
                    if not self.args.ablate_quantizer:
                        # Compute activation metrics
                        activation_metrics = compute_classification_metrics(
                            epoch_activation_labels, 
                            epoch_activation_preds,
                            split,
                            prefix = 'activation_',
                            split_in_name=False
                        )
                    else:
                        # Compute activation metrics
                        activation_metrics = compute_classification_metrics(
                            epoch_activation_labels, 
                            epoch_activation_preds,
                            split,
                            split_in_name=False
                        )
                    
                    epoch_metrics = {**epoch_metrics, **activation_metrics}
                    
                    # Add to Saver
                    for k, v in activation_metrics.items():
                        self.saver.add_scalar(f'{split}/{k}', v, epoch)
                    
                    # Update result metrics
                    for metric in epoch_metrics:
                        if metric not in result_metrics[split]:
                            result_metrics[split][metric] = [epoch_metrics[metric]]
                        else:
                            result_metrics[split][metric].append(epoch_metrics[metric])
                
                    # Plot last batch
                    if epoch % self.args.plot_every == 0 and split!='test':
                        if self.args.ablate_quantizer:
                            fig = None
                            #rec_pred tio cpu
                            #fig = plot_timeseries(batch['src'], batch['tgt'], rec_pred.permute(0, 2, 1, 3).reshape(bs, nvars, -1)[:, :, -self.args.forecast_window:], batch['activation_labels'], activation_preds)
                        else:
                            fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                        self.saver.add_plot(f'{split}/sample_plots', fig, epoch) 
                    
                # Add learning rate to Saver
                self.saver.add_scalar('lr', optimizer.param_groups[0]['lr'], epoch)
                
                if epoch == 0:
                    """
                    try:
                        # Add Codebook t-SNE to Saver
                        self.saver.add_tsne(model.codebook.embedding.weight.data.cpu(), quantizer.num_classes, epoch)
                    except:
                        self.saver.add_tsne(model.codebook.embedding.weight.data.cpu()[list(quantizer.level2class.keys()), :], quantizer.num_classes, epoch)
                    """
                
                if self.args.no_valid:
                    if epoch >= self.args.warmup_epochs:
                        saved = False
                        # Update best metrics
                        if result_metrics['test']['mae'][-1] < lowest_test_mae:
                            lowest_test_mae = result_metrics['test']['mae'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/lowest_test_mae', lowest_test_mae, epoch)

                        if result_metrics['test']['mse'][-1] < lowest_test_mse:
                            lowest_test_mse = result_metrics['test']['mse'][-1]
                            if not saved:
                                # save predictions
                                assert split == 'test', 'This implementation assumes that test is the last split'
                                #fig = plot_timeseries_causal_mask(rec_true, rec_pred, mask)
                                if self.args.ablate_quantizer:
                                    fig = None
                                    rec_pred = rec_pred.cpu()
                                    fig = plot_timeseries(batch['src'], batch['tgt'], rec_pred.permute(0, 2, 1, 3).reshape(bs, nvars, -1)[:, :, -self.args.forecast_window:], batch['activation_labels'], activation_preds)
                                else:
                                    fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                                self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                                saved = True
                        # Add to Saver
                        self.saver.add_scalar('test/lowest_test_mse', lowest_test_mse, epoch)

                        if result_metrics['test']['mase'][-1] < lowest_test_mase:
                            lowest_test_mase = result_metrics['test']['mase'][-1]
                            if not saved:
                                # save predictions
                                assert split == 'test', 'This implementation assumes that test is the last split'
                                if self.args.ablate_quantizer:
                                    fig = None
                                    rec_pred = rec_pred.cpu()
                                    fig = plot_timeseries(batch['src'], batch['tgt'], rec_pred.permute(0, 2, 1, 3).reshape(bs, nvars, -1)[:, :, -self.args.forecast_window:], batch['activation_labels'], activation_preds)
                                else:
                                    fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                                self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                                saved = True
                        # Add to Saver
                        self.saver.add_scalar('test/lowest_test_mase', lowest_test_mase, epoch)

                        if result_metrics['test']['smape'][-1] < lowest_test_smape:
                            lowest_test_smape = result_metrics['test']['smape'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/lowest_test_smape', lowest_test_smape, epoch)

                        if result_metrics['test']['accuracy'][-1] > max_test_acc:
                            max_test_acc = result_metrics['test']['accuracy'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/max_accuracy', max_test_acc, epoch)

                        if result_metrics['test']['balanced_accuracy'][-1] > max_test_balanced_acc:
                            max_test_balanced_acc = result_metrics['test']['balanced_accuracy'][-1]
                            if self.args.ablate_quantizer:
                                fig = None
                                rec_pred = rec_pred.cpu()
                                fig = plot_timeseries(batch['src'], batch['tgt'], rec_pred.permute(0, 2, 1, 3).reshape(bs, nvars, -1)[:, :, -self.args.forecast_window:], batch['activation_labels'], activation_preds)
                            else:
                                fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                            self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                        # Add to Saver
                        self.saver.add_scalar('test/max_balanced_accuracy', max_test_balanced_acc, epoch)
                
                else:
                    if epoch >= self.args.warmup_epochs:
                        saved = False
                        # Update best metrics
                        if result_metrics[val_name]['mae'][-1] < lowest_valid_mae:
                            lowest_valid_mae = result_metrics[val_name]['mae'][-1]
                            test_mae_at_lowest_valid_mae = result_metrics['test']['mae'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/mae_at_lowest_valid_mae', test_mae_at_lowest_valid_mae, epoch)
                        self.saver.add_scalar('valid/lowest_valid_mae', lowest_valid_mae, epoch)

                        if result_metrics[val_name]['mse'][-1] < lowest_valid_mse:
                            lowest_valid_mse = result_metrics[val_name]['mse'][-1]
                            test_mse_at_lowest_valid_mse = result_metrics['test']['mse'][-1]
                            # save the best model
                            self.saver.save_model(model, self.args.model, epoch, model_name=f"{self.args.model}_pretrained")
                            if not saved:
                                # save predictions
                                assert split == 'test', 'This implementation assumes that test is the last split'
                                fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                                self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                                saved = True
                        # Add to Saver
                        self.saver.add_scalar('test/mse_at_lowest_valid_mse', test_mse_at_lowest_valid_mse, epoch)
                        self.saver.add_scalar('valid/lowest_valid_mse', lowest_valid_mse, epoch)

                        if result_metrics[val_name]['mase'][-1] < lowest_valid_mase:
                            lowest_valid_mase = result_metrics[val_name]['mase'][-1]
                            test_mase_at_lowest_valid_mase = result_metrics['test']['mase'][-1]
                            if not saved:
                                # save predictions
                                assert split == 'test', 'This implementation assumes that test is the last split'
                                fig = plot_timeseries_causal_mask_quantized(src_patch, rec_true, rec_pred, mask, labels.permute(0, 2, 1), preds.permute(0, 2, 1), most_frequent_class)
                                self.saver.add_plot(f'{split}/sample_plots', fig, epoch)
                                saved = True
                        # Add to Saver
                        self.saver.add_scalar('test/mase_at_lowest_valid_mase', test_mase_at_lowest_valid_mase, epoch)
                        self.saver.add_scalar('valid/lowest_valid_mase', lowest_valid_mase, epoch)

                        if result_metrics[val_name]['smape'][-1] < lowest_valid_smape:
                            lowest_valid_smape = result_metrics[val_name]['smape'][-1]
                            test_smape_at_lowest_valid_smape = result_metrics['test']['smape'][-1]
                        # Add to Saver
                        self.saver.add_scalar('test/smape_at_lowest_valid_smape', test_smape_at_lowest_valid_smape, epoch)
                        self.saver.add_scalar('valid/lowest_valid_smape', lowest_valid_smape, epoch)

                if result_metrics['train']['loss'][-1] < lowest_train_loss:
                    lowest_train_loss = result_metrics['train']['loss'][-1]
                    if self.args.no_valid:
                        self.saver.save_model(model, self.args.model, epoch, model_name=f"{self.args.model}_pretrained")

                # Add to Saver
                self.saver.add_scalar('train/lowest_train_loss', lowest_train_loss, epoch)

                # Log all metrics
                self.saver.log()
                # Update scheduler
                if self.args.warmup_epochs > 0:
                    scheduler.step()
                elif self.args.use_scheduler:
                    scheduler.step(lowest_test_mse if self.args.no_valid else lowest_valid_mse)
                
                # Save model
                if epoch % self.args.save_every == 0:
                    self.saver.save_model(model, self.args.model, epoch, model_name=f"{self.args.model}_pretrained")

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass
        
        except FloatingPointError as err:
            print(f"Error: {err}")

        if self.args.no_valid:
            # print main metrics
            print(f"Lowest Test MAE: {lowest_test_mae}")
            print(f"Lowest Test MSE: {lowest_test_mse}")
            print(f"Highest Test Accuracy: {max_test_acc}")
            print(f"Highest Test Balanced Accuracy: {max_test_balanced_acc}")
        else:
            # print main metrics
            print(f"Lowest Valid MAE: {lowest_valid_mae}")
            print(f"Lowest Valid MSE: {lowest_valid_mse}")
            print(f"Test MAE at Lowest Valid MAE: {test_mae_at_lowest_valid_mae}")
            print(f"Test MSE at Lowest Valid MSE: {test_mse_at_lowest_valid_mse}")

        return model, result_metrics