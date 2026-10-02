from __future__ import annotations


import torch



from src.flynn.config import (
    LR, TRAIN_INPUT_SCALE, TRAIN_POLICY_HEAD, TRAIN_READOUT_HEAD, TRAIN_RETINA,
    TRAIN_RNN_BIAS, TRAIN_RNN_WEIGHTS, TRAIN_VALUE_HEAD, TRAIN_WIND_MLP,
)

def configure_optimizer(
    agent,
    lr: float = LR,
    train_rnn_weights: bool = TRAIN_RNN_WEIGHTS,
    train_rnn_bias: bool = TRAIN_RNN_BIAS,
    train_readout_head: bool = TRAIN_READOUT_HEAD,
    train_input_scale: bool = TRAIN_INPUT_SCALE,
    train_policy_head: bool = TRAIN_POLICY_HEAD,
    train_value_head: bool = TRAIN_VALUE_HEAD,
    train_wind_mlp: bool = TRAIN_WIND_MLP,
    train_retina: bool = TRAIN_RETINA,
) -> torch.optim.Optimizer:
    trainable_normal = []
    trainable_alpha = []
    
    print("[train] Configuring trainable parameters:")
    for name, p in agent.named_parameters():
        p.requires_grad = False
        train_this = False
        
        is_value_head = "value_head" in name
        is_policy_std = "policy_log_std" in name
        is_input_scale = "scale" in name and "input_scale" in name
        is_retina = "r_" in name or "l1" in name or "l2" in name or "l3" in name or "amacrine" in name
        
        is_cell = "cell" in name
        
        is_readout = is_cell and ("readout_head" in name or "head" in name)
        is_bias = is_cell and "bias" in name
        is_alpha = is_cell and "alpha" in name
        is_weight = is_cell and not (is_readout or is_bias or is_alpha)
        is_wind_mlp = "wind_mlp" in name

        if is_value_head and train_value_head:
            train_this = True
        elif is_policy_std and train_policy_head:
            train_this = True
        elif is_readout and train_readout_head:
            train_this = True
        elif is_input_scale and train_input_scale:
            train_this = True
        elif is_wind_mlp and train_wind_mlp:
            train_this = True
        elif is_retina and train_retina:
            train_this = True
        elif is_cell:
            if is_bias and train_rnn_bias:
                train_this = True
            elif is_alpha and (train_rnn_weights or train_rnn_bias): 
                train_this = True
            elif is_weight and train_rnn_weights:
                train_this = True
            
        if train_this:
            p.requires_grad = True
            if "alpha" in name:
                trainable_alpha.append(p)
                print(f"  [+] {name} (lr={lr*0.1:.2e})")
            else:
                trainable_normal.append(p)
                print(f"  [+] {name}")
        else:
            print(f"  [ ] {name}")
    
    param_groups = [{'params': trainable_normal, 'lr': lr}]
    if trainable_alpha:
        param_groups.append({'params': trainable_alpha, 'lr': lr * 0.1})
    
    optimizer = torch.optim.Adam(param_groups)
    print("[train] trainable normal params:", sum(p.numel() for p in trainable_normal))
    print("[train] trainable alpha params:", sum(p.numel() for p in trainable_alpha))
    
    return optimizer
