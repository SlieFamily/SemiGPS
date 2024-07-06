Config = {
    # global setting
    'dim_in': 138,
    'dim_out': 3,
    'node_enc_lst': ['LapPE', 'RWSE'],
    
    # position encoding
    'LapPE_cfg': {
        'dim_pe': 32,
        'dim_emb': 512,
        'model_type': 'DeepSet',
        'n_layers': 3,
        'post_n_layers': 1,
        'laplacian_norm': 'sym',
        # 'att_heads': 8, # if 'model_type' == 'Transformer'
        'max_freqs': 10,
        'norm_type': 'BatchNorm', # None
    },
    'RWSE_cfg': {
        'dim_pe': 32,
        'dim_emb': 512,
        'model_type': 'mlp', # 'linear'
        'n_layers': 3,
        'laplacian_norm': 'sym',
        'norm_type': 'BatchNorm', # None
        'ksteps': 10,
        'rw_ksteps': 10,
    },

    # SemiGPS model
    'local_gnn_type': 'CustomGatedGCN',
    'global_model_type': 'Transformer',
    'gps_layers': 5,
    'gt_dim': 512, # must be equal to 'dim_emb'
    'gt_heads': 10,
    'dropout': 0.0,
    'att_dropout': 0.0,
    'layer_norm': False, 
    'batch_norm': True,

    # train
    'alpha': 0.5, # loss factor
    'lr': 0.0006, 
    'lr_scheduler': 'step', 
    'lr_decay_steps':50, 
    'lr_decay_rate': 0.5, 
    'warmup_step': 100,

}