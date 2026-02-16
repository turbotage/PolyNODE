"""
Classification training script for spiral dynamics.

This module implements training and evaluation of neural ODE classifiers on spiral
trajectory data. The classifier learns to predict labels from encoded spiral trajectories
using a pretrained PolyNODE autoencoder for dimensionality reduction.

Key components:
    - Training loop with validation
    - Integration with reconstruction autoencoders
    - Support for multiple ODE solving methods (euler, rk4)
    - Configurable network architectures
    - Checkpoint saving and loading

Example:
    Train a classifier with default settings:
    ```bash
    python -m polynode.classification.classification --niters 1000 --lr 0.001
    ```
    
    Use a specific reconstruction model:
    ```bash
    python -m polynode.classification.classification \
        --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_2 \
        --niters 500
    ```
"""

import torch
import torch.nn as nn
import time
import math
import random
import numpy as np

from pathlib import Path
from datetime import datetime

from polynode.classification import classification_spiral_lib as class_spiral_lib
from polynode.classification.classification_spiral_lib import TrainingState

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

#torch.use_deterministic_algorithms(True)

# Setup organized directory structure
saves_root = Path(__file__).parent.parent.parent / "saves" / "classification"

plot = True

shared_data, dash_proc, args, hpara, X_ae, model_name = class_spiral_lib.setup()

train_loader, val_loader, t, t_latent = class_spiral_lib.get_dataloaders(hpara)

X_class = class_spiral_lib.get_classification_vf(hpara, args)

loss_func, eval_func = class_spiral_lib.get_loss_and_eval(X_class, X_ae, args, hpara)

optimizer = class_spiral_lib.get_optimizer(X_class, args)

lr_scheduler = class_spiral_lib.get_lr_scheduler(optimizer)


trainingstate = TrainingState(args, hpara)

for epoch in range(args.epoch):

    train_mse_loss_list = []
    train_accuracy_list = []
    for i, (x, y) in enumerate(train_loader):
        #print(f"Batch {i+1}: x shape: {x.shape}")

        loss_func_output = loss_func(x, y, t_latent)
        loss = loss_func_output[0]
        output = loss_func_output[1]
        accuracy = loss_func_output[2]
        mse_loss = loss_func_output[3]
        labels = loss_func_output[4]

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        train_mse_loss_list.append(mse_loss.item())
        train_accuracy_list.append(accuracy)

    trainingstate.train_mse_loss_list.append(sum(train_mse_loss_list)/len(train_mse_loss_list))
    trainingstate.train_accuracy_list.append(sum(train_accuracy_list)/len(train_accuracy_list))

    xv, yv = next(iter(val_loader))
    loss_func_output = loss_func(xv.detach(), yv.detach(), t_latent)
    val_loss = loss_func_output[0].item()
    val_output = loss_func_output[1]
    val_accuracy = loss_func_output[2]
    val_mse_loss = loss_func_output[3].item()
    val_labels = loss_func_output[4]

    trainingstate.val_mse_loss_list.append(val_mse_loss)
    trainingstate.val_accuracy_list.append(val_accuracy)

    if plot:
        class_spiral_lib.dash_plotting(
                            shared_data, 
                            hpara, 
                            trainingstate, 
                            (labels, val_labels), 
                            (output, val_output)
                        )

    if (len(trainingstate.val_accuracy_list) > 1) and \
       (trainingstate.val_accuracy_list[-1] > max(trainingstate.val_accuracy_list[:-1])):
        saving_model = True
        trainingstate.best_model_state = X_class.state_dict()  # Store in memory instead of saving to disk
        trainingstate.best_model_epoch = epoch
    else:
        saving_model = False

    current_lr = optimizer.param_groups[0]['lr']
    class_spiral_lib.print_stats(
                        epoch, 
                        current_lr, 
                        saving_model, 
                        trainingstate
                    )

    if class_spiral_lib.should_shutdown():
        break

    lr_scheduler.step(val_loss)

# Save the best model to organized directory structure
if trainingstate.best_model_state is not None:
    # Create directory structure: saves/classification/{reconstruction_model}/{timestamp}/
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = saves_root / args.reconstruction_model / timestamp
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\n{'='*70}")
    print(f"Training completed!")
    print(f"{'='*70}")
    print(f"Best model found at epoch {trainingstate.best_model_epoch}")
    print(f"Best validation accuracy: {max(trainingstate.val_accuracy_list):.4f}")
    
    trainingstate.save(save_dir)
    
    print(f"{'='*70}\n")

try:
    dash_proc.join()  # Keep the process running
except KeyboardInterrupt:
    dash_proc.terminate()


