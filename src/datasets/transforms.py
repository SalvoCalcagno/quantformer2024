import torch
import torch.nn as nn

# Transformations
class Standardize(object):
    """Standardize a tensor image with mean and standard deviation.
    """
    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor series of size (T) to be standardized.
        Returns:
            Tensor: Standardized Tensor image.
        """
        return (tensor - tensor.mean())/tensor.std()
    
class RandomFlipSign(object):
    """Flip the sign of a tensor series.
    """
    def __init__(self, flip_prob=0.5):
        self.flip_prob = flip_prob

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor series of size (T) to be flipped.
        Returns:
            Tensor: Flipped Tensor image.
        """
        if torch.rand(1) < self.flip_prob:
            return -tensor
        else:
            return tensor
    
class RandomReverse(object):
    """Reverse a tensor series.
    """
    def __init__(self, reverse_prob=0.5):
        self.reverse_prob = reverse_prob

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor series of size (T) to be reversed.
        Returns:
            Tensor: Reversed Tensor image.
        """
        if torch.rand(1) < self.reverse_prob:
            return torch.flip(tensor, dims=[0])
        else:
            return tensor
    
class RandomGaussianNoise():

    def __init__(self, p=0.5, noise=0.05):
        self.p = p
        self.noise = noise

    def __call__(self, tensor):
        if torch.rand(1) < self.p:
            return tensor + torch.randn(tensor.shape)*self.noise
        else:
            return tensor

class SeparateSourceAndTarget():
    """Separate source and target from a tensor series.
    """
    def __init__(self, source_length, multivariate=False):
        self.source_length = source_length
        self.multivariate = multivariate

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor series of size (T) to be separated.
        Returns:
            Tensor: Source Tensor.
            Tensor: Target Tensor.
            Tensor: Target_y Tensor
        """
        if self.multivariate:
            src = tensor[:self.source_length]
            tgt = tensor[self.source_length-1:-1]
            tgt_y = tensor[self.source_length:]
        else:
            src = tensor[:self.source_length].unsqueeze(-1)
            tgt = tensor[self.source_length-1:-1].unsqueeze(-1)
            tgt_y = tensor[self.source_length:].unsqueeze(-1)
        return src, tgt, tgt_y

class TraceOnly():
    """Remove all keys from the dict, but trace.
    """
    def __call__(self, dict):
        """
        Args:
            dict (dict): Dict to be removed.
        Returns:
            tensor: trace only.
        """
        if 'class' in dict:
            return dict['trace'], dict['class']
        return dict['trace']
    
class PrepareInformerInput():
    """Prepare input for Informer model.
    """
    def __init__(self, source_length, context_length):
        """
        This transformation is used to prepare input for Informer model.
        Informer encoder receives a sequence (SOURCE) of source_length tokens as input.
        Informer decoder receives a sequence (CONTEXT|TARGET) of context_length+target_length tokens as input.
        CONTEXT contains the last context_length tokens of SOURCE.
        TARGET contains target_length tokens to be predicted after CONTEXT. This part will be used as ground truth.
        The trainer will be responsible to fill the TARGET input with padding (0 or 1).

        Args:
            source_length (int): Length of source sequence.
            context_length (int): Length of context sequence.
        """
        self.source_length = source_length
        self.context_length = context_length
        #self.target_length = target_length

    def __call__(self, x_dict):
        """
        Args:
            x_dict (dict): Tensor series of size (T, num_features) to be separated.
        Returns:
            Tensor: Source Tensor.
            Tensor: Context Tensor.
            Tensor: Target Tensor
        """
        trace = x_dict['trace']
        #time = x_dict['time']
        #time = torch.tensor(x_dict['s_ids'])/160
        time = torch.tensor(x_dict['s_embeddings'])

        x = trace[:self.source_length]
        x_mark = time[:self.source_length]
        y = trace[self.source_length-self.context_length:]
        y_mark = time[self.source_length-self.context_length:]

        return x, x_mark, y, y_mark
    
class MaxPool():

    def __init__(self, kernel_size=2):
        self.kernel_size = kernel_size
        self.pool = nn.MaxPool1d(kernel_size=kernel_size)

    def __call__(self, x):
        return self.pool(torch.tensor(x))

class PreparePatchTSTInput():

    def __init__(self, source_length):
        
        self.source_length = source_length

    def __call__(self, x):
        # x is a tensor of shape (total_len, num_features)
        # we want to produce src and tgt tensors of shape 
        # src --> (num_features, source_length)
        # tgt --> (num_features, total_len - source_length)

        #x = x.T
        class_label = None
        if type(x) == tuple:
            x, class_label = x
        src = torch.tensor(x[:self.source_length, :]).float()
        tgt = torch.tensor(x[self.source_length:, :]).float()
        #print(f"class_label: {class_label}")

        return src, tgt, torch.tensor(class_label, dtype=torch.long)
    
class PrepareCrossformerInput():

    def __init__(self, source_length):
        
        self.source_length = source_length

    def __call__(self, x):
        # x is a tensor of shape (total_len, num_features)
        # we want to produce src and tgt tensors of shape 
        # src --> (num_features, source_length)
        # tgt --> (num_features, total_len - source_length)

        src = torch.tensor(x[:self.source_length, :]).float()
        tgt = torch.tensor(x[self.source_length:, :]).float()

        return src, tgt