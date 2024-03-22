multilog_config = {
    'model_name': 'multilog',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'mask_rate': 0.15,
    'top_k': 1,
    'reduction_dim': 32,
    'pretrained_model': 'multilingual_mini',
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'dropout': 0.1,
    'ff_dim': 1024,
    'num_heads': 6,
    'num_blocks': 2,
    'epochs': 50,
    'verbose': 1,
    'generator_batch_size': 512,
    'loss_weights': {
        'mask_out': 0.5,
        'next_out': 0.4,
        'hypersphere': 0.1
    }
}

pn_bert_config = {
    'model_name': 'pn_bert',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'mask_rate': 0.15,
    'top_k': 1,
    'pretrained_model': 'multilingual_mini',
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'dropout': 0.1,
    'ff_dim': 1024,
    'num_heads': 6,
    'num_blocks': 2,
    'epochs': 50,
    'verbose': 1,
    'generator_batch_size': 512,
    'loss_weights': {
        'mask_out': 0.5,
        'next_out': 0.5
    },
    'reduction_dim': 32,
}

n_bert_config = {
    'model_name': 'n_bert',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'mask_rate': 0.15,
    'top_k': 1,
    'pretrained_model': 'multilingual_mini',
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'dropout': 0.1,
    'ff_dim': 1024,
    'num_heads': 6,
    'num_blocks': 2,
    'epochs': 50,
    'verbose': 1,
    'generator_batch_size': 512,
    'reduction_dim': 32,
}

logbert_config = {
    'model_name': 'logbert',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'mask_rate': 0.65,
    'top_k': 1,
    'embed_size': 256,
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'dropout': 0.1,
    'num_heads': 4,
    'num_blocks': 4,
    'ff_dim': 256,
    'epochs': 200,
    'verbose': 1,
    'loss_weights': {
        'mask_out': 1,
        'hypersphere': 0.1
    },
    'generator_batch_size': 512
}

deeplog_config = {
    'model_name': 'deeplog',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'hidden_size': 128,
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'epochs': 100,
    'verbose': 1,
    'generator_batch_size': 512
}

loganomaly_config = {
    'model_name': 'loganomaly',
    'window_size': 10,
    'fixed_window': 20,
    'val_pos': 'head',
    'shuffle': True,
    'random_seed': 50,
    'hidden_size': 128,
    'model_file': 'best_model.hdf5',
    'batch_size': 512,
    'epochs': 50,
    'verbose': 1,
    'use_semantic': True,
    'pretrained_model': 'fast_text',
    'reduction_dim': 300,
    'generator_batch_size': 512
}
