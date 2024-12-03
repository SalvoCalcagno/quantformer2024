import pkgutil

__all__ = []
for loader, module_name, is_pkg in  pkgutil.walk_packages(__path__):
    if is_pkg or "." in module_name:
        continue
    if module_name in [
        'eegnet',
        'syncnet',
        'eegchannelnet',
        'rnn',
        'cnn',
        'autoregressive_lstm',
        'tft_transformer',
        'informer',
        'patchtst',
        'crossformer',
        'patch_vqvae',
        'quantformer',
        'crossquantformer',
        'crossquantcoder',
        'quantcoder',
        'lstm_classification',
        'brain_transformer',
        'patchtst_pretrain',
        'patchtst_finetune',
        'patchtst_finetune_nochans',
        'autoformer',]:
        
        __all__.append(module_name)
        _module = loader.find_module(module_name).load_module(module_name)
        globals()[module_name] = _module