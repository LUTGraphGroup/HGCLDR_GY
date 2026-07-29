import argparse
from utils.util import add_flags_from_config

config_args = {
    'training_config': {
        'log': ('cd', 'None for no logging'),
        'lr': (0.001, 'learning rate'),
        'weight-decay': (0.005, 'l2 regularization strength'),
        'momentum': (0.95, 'momentum in optimizer'),
        'epochs': (1000, 'maximum number of epochs to train for'),
        'batch-size': (512, 'batch size'),
        'seed': (1234, 'seed for data split and training'),
        'log-freq': (1, 'how often to compute print train/val metrics (in epochs)'),
        'eval-freq': (10, 'how often to compute val metrics (in epochs)'),
        'eval_batch_num': (0, 'val compute batch num'),
        'selection_only': (0, 'stop after validation-based model selection without accessing outer test data'),
        'use_fixed_validation_pairs': (0, 'use versioned fixed validation pairs for model selection'),
        'device': ('cuda:0', 'device'),
        'refit_after_selection': (0, 'retrain on outer train plus validation after model selection'),
        'refit_seed_offset': (1000000, 'seed offset for the outer-fold refit phase'),
        'output_root': ('results', 'root directory for experiment artifacts'),
        'run_tag': ('main_none', 'subdirectory used to isolate experiment variants')
    },
    'model_config': {
        'embedding_dim': (32, 'drug disease embedding dimension'),
        'network': ('resSumGCN', 'choice of StackGCNs, plainGCN, denseGCN, resSumGCN, resAddGCN'),
        'scale': (0.1, 'scale'),
        'max_norm': (1.5, 'max norm'),
        'num-layers': (4,  'number of hidden layers in encoder'),
        'margin': (0.1, 'margin value in the metric learning loss'),
        'ssl_ratio': (0.05, ''),
        'ssl_temp': (0.05, 'ssl temp'),
        'ssl_reg': (0.005, 'ssl reg'),
        'structure_aug': ('True', 'enable structure augmentation with auxiliary similarity graphs')
    },
    'data_config': {
        'dataset': ('B-dataset', 'which dataset to use'),
        'fold': (None, 'required outer cross-validation fold index (0-9)'),
        'pseudo_mode': ('none', 'pseudo supervision: none, hard, or weighted'),
        'pseudo_pos_fraction': (0.2, 'pseudo-positive rows relative to observed rows'),
        'pseudo_confidence_threshold': (0.0, 'minimum normalized pseudo confidence'),
        'num_neg': (8, 'number of negative samples'),
        'norm_adj': ('True', 'whether to row-normalize the adjacency matrix'),
    }
}

parser = argparse.ArgumentParser()
for _, config_dict in config_args.items():
    parser = add_flags_from_config(parser, config_dict)
