import torch
import numpy as np
from tqdm import tqdm
import torch.optim as optim
import src.models as models
from src.saver import Saver
import torch.nn.functional as F
from src.utils import make_ts_grid, compare_traces
from torch.utils.data import DataLoader
import matplotlib.gridspec as gridspec
from src.models.patchtst_self import Patch, PatchMask
from torchvision.utils import make_grid
from matplotlib import pyplot as plt
from src.metrics.correlation import TimeSeriesPearsonCorrCoef
from src.metrics.FocalLoss import FocalLoss
import torch.nn as nn
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve, auc
from src.utils import plot_timeseries_causal_mask
from src.metrics.PretrainLoss import MaskedMSELoss

class Trainer():
    
    def __init__(self, args):
        
        # Store Args
        self.args = args

        # Set criterion
        # Criterion is not initialized here, but in the train method
        # It is fixed for VQVAE and it is the combination of 3 MSE losses
    
        # Set Saver
        self.saver = Saver(args.logdir, args.exp_name)
        # Save args
        self.saver.save_configuration(vars(args))
        
    def train(self, datasets):
        return self.train_quantizer(datasets)
        
    def train_quantizer(self, datasets):

        splits = list(datasets.keys())
        
        def collate_fn_eval(batch_l):
                
            # get number of neurons per sample
            #num_neurons = [sample['src'].shape[0] for sample in batch_l]
            
            # expand stimulus
            #stim = [torch.tensor(sample['stim']).expand(n, -1) for sample, n in zip(batch_l, num_neurons)]
            
            # init batch
            batch = {
                'src': torch.vstack([torch.tensor(sample['src']) for sample in batch_l]),
                'tgt': torch.vstack([torch.tensor(sample['tgt']) for sample in batch_l]),
                'activation_labels': torch.cat([torch.tensor(sample['activation_labels']) for sample in batch_l]),
                #'stim': torch.vstack([torch.tensor(s) for s in stim]),
            }
            
            return batch
        
        def collate_fn(batch_l):
            
            #batch_l is the list of samples
            batch = {}
            num_neurons = [sample['src'].shape[0] for sample in batch_l]
            #stim = [torch.tensor(sample['stim']).expand(n, -1) for sample, n in zip(batch_l, num_neurons)]
            
            batch['src'] = torch.vstack([torch.tensor(sample['src']) for sample in batch_l])
            batch['tgt'] = torch.vstack([torch.tensor(sample['tgt']) for sample in batch_l])
            batch['activation_labels'] = torch.cat([torch.tensor(sample['activation_labels']) for sample in batch_l])
            #batch['stim'] = torch.vstack([torch.tensor(sample) for sample in stim])
                
            # select all activations
            active_mask = batch['activation_labels'].bool()
            active_src = batch['src'][active_mask]
            active_tgt = batch['tgt'][active_mask]
            #active_stim = batch['stim'][active_mask]
            active_labels = batch['activation_labels'][active_mask]
            
            # negative samples
            inactive_mask = ~active_mask
            inactive_src = batch['src'][inactive_mask]
            inactive_tgt = batch['tgt'][inactive_mask]
            #inactive_stim = batch['stim'][inactive_mask]
            inactive_labels = batch['activation_labels'][inactive_mask]
            
            # shuffle negative samples
            idx = torch.randperm(inactive_src.shape[0])
            inactive_src = inactive_src[idx]
            inactive_tgt = inactive_tgt[idx]
            #inactive_stim = inactive_stim[idx]
            inactive_labels = inactive_labels[idx]
            
            # select same number of negative samples
            n = active_src.shape[0]
            inactive_src = inactive_src[:n]
            inactive_tgt = inactive_tgt[:n]
            #inactive_stim = inactive_stim[:n]
            inactive_labels = inactive_labels[:n]
            
            # merge active and inactive samples
            batch['src'] = torch.vstack([active_src, inactive_src])
            batch['tgt'] = torch.vstack([active_tgt, inactive_tgt])
            #batch['stim'] = torch.vstack([active_stim, inactive_stim])
            batch['activation_labels'] = torch.cat([active_labels, inactive_labels])

            return batch
        
        def collate_fn_chs(batch_l):
            
            # get number of neurons per sample
            num_neurons = [sample['src'].shape[0] for sample in batch_l]
            src_len = batch_l[0]['src'].shape[1]
            tgt_len = batch_l[0]['tgt'].shape[1]
            
            # max number of neurons
            max_neurons = max(num_neurons)
            
            src = torch.zeros(len(batch_l), max_neurons, src_len) -100
            tgt = torch.zeros(len(batch_l), max_neurons, tgt_len) -100
            activation_labels = torch.zeros(len(batch_l), max_neurons) -100
            
            for i, sample in enumerate(batch_l):
                src[i, :num_neurons[i]] = torch.tensor(sample['src'])
                tgt[i, :num_neurons[i]] = torch.tensor(sample['tgt'])
                activation_labels[i, :num_neurons[i]] = torch.tensor(sample['activation_labels'])
            
            batch = {
                'src': src,
                'tgt': tgt,
                'activation_labels': activation_labels,
            }

            return batch
        
        if self.args.model == 'crossquantformer':
            # Create loaders
            loaders = {
                split: DataLoader(
                    datasets[split],
                    batch_size=self.args.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.args.workers,
                    collate_fn=collate_fn_chs,
                ) for split in splits
            }
        else:
            # Create loaders
            loaders = {
                split: DataLoader(
                    datasets[split],
                    batch_size=self.args.batch_size,
                    shuffle=(split == 'train'),
                    num_workers=self.args.workers,
                    collate_fn=collate_fn if split == 'train' else collate_fn_eval,
                ) for split in splits
            }

        # Create model
        module = getattr(models, self.args.model)
        model = getattr(module, "Model")(vars(self.args))
        
        # Move to device
        model = model.to(self.args.device)

        optimizer = torch.optim.Adam(model.parameters(), lr=self.args.lr)

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
        
        # Create patching operations
        self.patch = Patch(
            self.args.source_length,
            self.args.patch_len,
            self.args.patch_len,
        )
        
        self.masked_patch = PatchMask(
            self.args.patch_len,
            self.args.patch_len,
            self.args.mask_ratio,
            force_causal_masking=self.args.force_causal_masking,
            mask_on_peaks=self.args.mask_on_peaks
        )

        # Initialize the final result metrics
        result_metrics = { split: {} for split in splits }
        corr = TimeSeriesPearsonCorrCoef()
        
        masked_mse_loss = MaskedMSELoss()

        # Test Metrics
        lowest_test_loss = float('inf')

        # Train Metrics
        lowest_train_loss = float('inf')
        
        if 'valid' in splits or 'val' in splits:
            raise NotImplementedError('This implementation assumes that there is no validation set')
        
        # Plot the first batch of the test set
        fixed_traces = next(iter(loaders['test']))
        fixed_traces_src = fixed_traces['src']
        fixed_traces_tgt = fixed_traces['tgt']
        fixed_traces = torch.cat((fixed_traces_src, fixed_traces_tgt), dim=-1)
        if self.args.model != 'crossquantformer':
            if self.args.dataset in ['pretrain', 'cross_stimuli', 'cross_stimuli2', 'cross_session', 'cross_subject', 'foundation']:
                fixed_traces = fixed_traces.unsqueeze(1)
        # select random 32 traces
        torch.manual_seed(42)
        idx = torch.randperm(fixed_traces.shape[0])[:32]
        fixed_traces = fixed_traces.transpose(1, 2)[idx]#, 0]
        # fixed_grid = make_grid(fixed_traces.unsqueeze(1), nrow=8, range=(-1, 1), normalize=True)
        #neuron = torch.randint(0, fixed_traces.shape[2], (1,)).item()
        #fixed_series = fixed_traces[:, :, neuron]
        #fixed_ts_grid = make_ts_grid(fixed_series, nrow=8)
        
        #self.saver.add_images('original_traces', fixed_grid, 0)
        #self.saver.add_plot('original_ts', fixed_ts_grid, 0)
        
        fixed_traces = fixed_traces.to(self.args.device)
        x_tilde, z_e_x, z_q_x, _ = model(self.patch(fixed_traces))
        x_tilde = x_tilde.to('cpu')
        bs, num_patch, nvars, patch_len = x_tilde.shape
        
        reconstructed_traces = x_tilde.permute(0, 2, 1, 3).reshape(bs, nvars, -1).permute(0, 2, 1)
        #reconstructed_grid = make_grid(reconstructed_traces.unsqueeze(1), nrow=8, range=(-1, 1), normalize=True)

        #reconstructed_series = reconstructed_traces[:, :, neuron]
        #reconstructed_ts_grid = make_ts_grid(reconstructed_series.detach(), nrow=8)
        
        #self.saver.add_images('reconstructed_traces', reconstructed_grid, 0)
        #self.saver.add_plot('reconstructed_ts', reconstructed_ts_grid, 0)
        plot = compare_traces(fixed_traces, reconstructed_traces)
        self.saver.add_plot('reconstructed_traces', plot, 0)
        
        def mse_loss(input, target, ignored_index, reduction):
            mask = target == ignored_index
            out = (input[~mask]-target[~mask])**2
            if reduction == "mean":
                return out.mean()
            elif reduction == "None":
                return out
        
        def train_epoch_masked(data_loader, model, optimizer, epoch):
    
            model.train()
            
            losses_recons = []
            losses_vq = []
            losses_commit = []
            losses_masked_recons = []
            
            correlations = []
            patch_correlations = []
            used_codes = []
            
            for batch in tqdm(data_loader):
                
                # Patch the data
                x = torch.cat((batch['src'], batch['tgt']), -1)
                if self.args.model != 'crossquantformer':
                    if self.args.dataset in ['pretrain', 'cross_stimuli', 'cross_stimuli2', 'cross_session', 'cross_subject', 'foundation']:
                        x = x.unsqueeze(1)
                x = x.transpose(1, 2)
                x_patch, x_mask, mask = self.masked_patch(x)
                
                # Move to device
                x_patch = x_patch.to(self.args.device)
                x_mask = x_mask.to(self.args.device)
                mask = mask.to(self.args.device)
                #stim_id = stim_id.to(self.args.device)
                stim_id=None
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                x_tilde, z_e_x, z_q_x, ids = model(x_mask, stimulus_id=stim_id, mask=mask)
                ids = ids['q']

                # Compute loss
                
                # Reconstruction loss
                #loss_recons = F.mse_loss(x_tilde, x_patch)
                loss_recons = mse_loss(x_tilde, x_patch, -100, "mean")
                # Masked MSE loss
                pad_mask = x_mask == -100
                loss_masked_recons = masked_mse_loss(x_tilde, x_patch, ~pad_mask[:, :, :, 0] & mask)           
                if self.args.ablate_quantizer:
                    # We only use the reconstruction loss
                    loss = loss_recons + loss_masked_recons
                    loss_vq = torch.tensor(0)
                    loss_commit = torch.tensor(0)
                else:
                    # Vector quantization objective
                    loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                    # Commitment objective
                    loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                    # Total loss
                    loss = loss_recons + loss_vq + self.args.beta * loss_commit + loss_masked_recons

                
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Store losses
                losses_recons.append(loss_recons.item())
                losses_vq.append(loss_vq.item())
                losses_commit.append(loss_commit.item())
                losses_masked_recons.append(loss_masked_recons.item())
                
                # Compute correlation
                correlation = corr(x_tilde, x_patch, patched=False)
                patch_correlation = corr(x_tilde, x_patch, patched=True)
                
                # Store correlation
                correlations.append(correlation.item())
                patch_correlations.append(patch_correlation.item())
                
                # Compute used codes
                uc = (ids.unique().shape[0])/self.args.K
                
                # Store used codes
                used_codes.append(uc)
            
            loss_recons = np.mean(np.array(losses_recons))
            loss_vq = np.mean(np.array(losses_vq))
            loss_commit = np.mean(np.array(losses_commit))
            loss_masked_recons = np.mean(np.array(losses_masked_recons))
            
            correlations = np.mean(np.array(correlations))
            patch_correlations = np.mean(np.array(patch_correlations))
            
            used_codes = np.mean(np.array(used_codes))
            
            # Logs
            self.saver.add_scalar('train/reconstruction_loss', loss_recons.item(), epoch)
            self.saver.add_scalar('train/quantization_loss', loss_vq.item(), epoch)
            self.saver.add_scalar('train/commitment_loss', loss_commit.item(), epoch)
            self.saver.add_scalar('train/masked_reconstruction_loss', loss_masked_recons.item(), epoch)
            
            self.saver.add_scalar('train/correlation', correlations.item(), epoch)
            self.saver.add_scalar('train/patch_correlation', patch_correlations.item(), epoch)
            
            self.saver.add_scalar('train/used_codes', used_codes.item(), epoch)
            
            return loss_recons.item() + loss_vq.item(), correlations.item()

        def train_epoch(data_loader, model, optimizer, epoch):
    
            model.train()
            
            losses_recons = []
            losses_vq = []
            losses_commit = []
            
            correlations = []
            patch_correlations = []
            
            used_codes = []
            
            for batch in tqdm(data_loader):
                
                # Patch the data
                x = torch.cat((batch['src'], batch['tgt']), -1)
                if self.args.dataset in ['pretrain', 'cross_stimuli', 'cross_stimuli2', 'cross_session', 'cross_subject', 'foundation']:
                    x = x.unsqueeze(1)
                x = x.transpose(1, 2)
                x_patch = self.patch(x)
                
                # Move to device
                x_patch = x_patch.to(self.args.device)
                #stim_id = stim_id.to(self.args.device)
                stim_id=None
                
                # Reset gradients
                optimizer.zero_grad()
                
                # Forward pass
                x_tilde, z_e_x, z_q_x, ids = model(x_patch, stimulus_id=stim_id, mask=None)
                ids = ids['q']

                # Compute loss
                
                # Reconstruction loss
                loss_recons = F.mse_loss(x_tilde, x_patch)
                # Vector quantization objective
                loss_vq = F.mse_loss(z_q_x, z_e_x.detach())
                # Commitment objective
                loss_commit = F.mse_loss(z_e_x, z_q_x.detach())
                # Total loss
                loss = loss_recons + loss_vq + self.args.beta * loss_commit
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                # Store losses
                losses_recons.append(loss_recons.item())
                losses_vq.append(loss_vq.item())
                losses_commit.append(loss_commit.item())
                
                # Compute correlation
                correlation = corr(x_tilde, x_patch, patched=False)
                patch_correlation = corr(x_tilde, x_patch, patched=True)
                
                # Store correlation
                correlations.append(correlation.item())
                patch_correlations.append(patch_correlation.item())
                
                # Compute used codes
                uc = (ids.unique().shape[0])/self.args.K
                
                # Store used codes
                used_codes.append(uc)
            
            loss_recons = np.mean(np.array(losses_recons))
            loss_vq = np.mean(np.array(losses_vq))
            loss_commit = np.mean(np.array(losses_commit))
            
            correlations = np.mean(np.array(correlations))
            patch_correlations = np.mean(np.array(patch_correlations))
            
            used_codes = np.mean(np.array(used_codes))
            
            # Logs
            self.saver.add_scalar('train/reconstruction_loss', loss_recons.item(), epoch)
            self.saver.add_scalar('train/quantization_loss', loss_vq.item(), epoch)
            self.saver.add_scalar('train/commitment_loss', loss_commit.item(), epoch)
            
            self.saver.add_scalar('train/correlation', correlations.item(), epoch)
            self.saver.add_scalar('train/patch_correlation', patch_correlations.item(), epoch)
            
            self.saver.add_scalar('train/used_codes', used_codes.item(), epoch)
            
            return loss_recons.item() + loss_vq.item(), correlations.item()
        
        def test_epoch(data_loader, model, epoch):
            
            model.eval()
        
            with torch.no_grad():
                loss_recons, loss_vq = 0., 0.
                correlation, patch_correlation = 0., 0.
                used_codes = []
                for batch in tqdm(data_loader):
                    
                    # Patch the data
                    x = torch.cat((batch['src'], batch['tgt']), -1)
                    if self.args.model != 'crossquantformer':
                        if self.args.dataset in ['pretrain', 'cross_stimuli', 'cross_stimuli2', 'cross_session', 'cross_subject', 'foundation']:
                            x = x.unsqueeze(1)
                    x = x.transpose(1, 2)
                    x = self.patch(x)
                    
                    # Move to device
                    x = x.to(self.args.device)
                    #stim_id = stim_id.to(self.args.device)
                    stim_id = None
                    
                    # Forward pass
                    x_tilde, z_e_x, z_q_x, ids = model(x, stimulus_id=stim_id, mask=None)
                    ids = ids['q']
                    
                    # Compute losses
                    loss_recons += F.mse_loss(x_tilde, x)
                    loss_vq += F.mse_loss(z_q_x, z_e_x)
                    
                    # Compute correlation
                    correlation += corr(x_tilde, x, patched=False)
                    patch_correlation += corr(x_tilde, x, patched=True)
                    
                    # Compute used codes
                    uc = (ids.unique().shape[0])/self.args.K
                    
                    # Store used codes
                    used_codes.append(uc)

                # Average losses
                loss_recons /= len(data_loader)
                loss_vq /= len(data_loader)
                
                # Average correlation
                correlation /= len(data_loader)
                patch_correlation /= len(data_loader)
                
                # Average used codes
                used_codes = np.mean(np.array(used_codes))

            # Logs
            self.saver.add_scalar('test/reconstruction_loss', loss_recons.item(), epoch)
            self.saver.add_scalar('test/quantization_loss', loss_vq.item(), epoch)
            
            self.saver.add_scalar('test/correlation', correlation.item(), epoch)
            self.saver.add_scalar('test/patch_correlation', patch_correlation.item(), epoch)
            
            self.saver.add_scalar('test/used_codes', used_codes.item(), epoch)
            
            return loss_recons.item() + loss_vq.item(), correlation.item()

        # Train the model
        best_loss = float('inf')
        best_correlation = 0.
        
        try:
            for epoch in range(1, self.args.epochs+1):

                # Train step
                if self.args.mask_ratio == 0:
                    train_loss, train_correlation = train_epoch(loaders['train'], model, optimizer, epoch)
                else:
                    train_loss, train_correlation = train_epoch_masked(loaders['train'], model, optimizer, epoch)
                
                # Evaluate step
                test_loss, test_correlation = test_epoch(loaders['test'], model, epoch)
                
                # Use model in inference mode
                model.eval()
                
                x_tilde, _, _, _ = model(self.patch(fixed_traces))
                x_tilde = x_tilde.to('cpu')
                bs, num_patch, nvars, patch_len = x_tilde.shape
        
                reconstructed_traces = x_tilde.permute(0, 2, 1, 3).reshape(bs, nvars, -1).permute(0, 2, 1)
                #reconstructed_grid = make_grid(reconstructed_traces.unsqueeze(1), nrow=8, range=(-1, 1), normalize=True)

                #reconstructed_series = reconstructed_traces[:, :, neuron]
                #reconstructed_ts_grid = make_ts_grid(reconstructed_series.detach(), nrow=8)
                
                #self.saver.add_images('reconstructed_traces', reconstructed_grid, epoch)
                #self.saver.add_plot('reconstructed_ts', reconstructed_ts_grid, epoch)
                plot = compare_traces(fixed_traces, reconstructed_traces)
                self.saver.add_plot('reconstructed_traces', plot, epoch)
                
                # Save metrics
                self.saver.add_scalar('train/loss', train_loss, epoch)
                self.saver.add_scalar('test/loss', test_loss, epoch)
                
                # Update scheduler
                if scheduler is not None:
                    scheduler.step(lowest_test_loss)
                
                # Save model
                if train_correlation > best_correlation:
                    best_correlation = train_correlation
                    self.saver.save_model(model, self.args.model, epoch=epoch)
                    self.saver.add_scalar('train/best_correlation', best_correlation, epoch)
                
                if train_loss < train_loss:
                    best_loss = train_loss
                #    self.saver.save_model(model, self.args.model, epoch=epoch, model_name=f"best_model")
                
                # Log all metrics
                self.saver.log()

        except KeyboardInterrupt:
            print('KeyboardInterrupt')
            pass
        
        except FloatingPointError as err:
            print(f"Error: {err}")
            
        print("Self-supervised training finished")
        
        return model, result_metrics