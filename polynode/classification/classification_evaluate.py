"""
Evaluation utilities for trained classification models.

This module provides functions to evaluate classification performance on test data,
load trained models from organized directory structures, generate static plots,
and optionally launch interactive visualizations.

Uses existing library functions:
    - TrainingState.load() for loading saved models
    - Autoencoder loading approach from spiral_lib.setup()
    
Key components:
    - find_run_directory: Locate a specific training run
    - load_model_from_config: Load model using library functions
    - plot_trajectories_eval: Generate and save matplotlib trajectory plots
    - list_available_models: Discover all trained models
    - Command-line interface for model evaluation and visualization

Default behavior:
    - Loads model and displays configuration and final metrics
    - Generates trajectory plots (train and validation) 
    - Generates loss curve plots
    - Saves all plots as .png and .eps files in the model's run directory
    
Optional features:
    - Use --dash flag to launch interactive Plotly visualization server
    - Use --skip_plots to only display stats without generating plots
    
Command line usage:
    ```bash
    # List all available models
    python -m polynode.classification.classification_evaluate --list_models
    
    # Evaluate most recent run (generates and saves plots)
    python -m polynode.classification.classification_evaluate \\
        --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1
    
    # Evaluate specific run
    python -m polynode.classification.classification_evaluate \\
        --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1 \\
        --run_id 20260209_143022
    
    # Launch interactive Dash app for visualization
    python -m polynode.classification.classification_evaluate \\
        --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1 \\
        --dash
    ```
"""

import torch
import torch.nn as nn
import time
import math
import numpy as np
import random
import json
import argparse

import matplotlib.pyplot as plt

from pathlib import Path
from torchdiffeq import odeint

import polynode.plotly_plot as pp
from polynode.classification import classification_spiral_lib as class_spiral_lib
from polynode.classification.classification_spiral_lib import TrainingState

seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def plot_trajectories_eval(trajectories, labels=None, title="Trajectories", target_points=None, target_labels=None, ax_view=None, filepath=None):
    """
    Plot 2D or 3D trajectories using matplotlib, including points and optional target points.

    """
    timepoints, batchsize, dims = trajectories.shape
    if labels is None:
        labels = [f'Traj {i+1}' for i in range(batchsize)]
    labels = [str(l) for l in labels]
    if target_labels is not None:
        target_labels = [str(l) for l in target_labels]

    # Build color map for all labels
    all_labels = set(target_labels) if target_labels is not None else set(labels)
    if target_labels is not None:
        all_labels.update(target_labels)
    color_map = plt.get_cmap('tab10')
    label_to_color = {label: color_map(i % 10) for i, label in enumerate(sorted(all_labels))}

    fig = plt.figure()
    if dims == 2:
        ax = fig.add_subplot(111)
        for i in range(batchsize):
            traj = trajectories[:, i, :]
            color = label_to_color[labels[i]]
            ax.plot(traj[:, 0], traj[:, 1], '-', color=color, label=f'Trajectory {labels[i]}')
            ax.plot(traj[:, 0], traj[:, 1], '.', color=color, markersize=3)
        # Plot target points
        if target_points is not None and target_labels is not None:
            for target, target_label in zip(target_points, target_labels):
                color = label_to_color[target_label]
                ax.plot(target[0], target[1], 'd', color=color, markersize=8, label=f'Target {target_label}')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
    elif dims == 3:
        ax = fig.add_subplot(111, projection='3d')
        for i in range(batchsize):
            traj = trajectories[:, i, :]
            color = label_to_color[labels[i]]
            ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '-', color=color, linewidth=1.5)#, label=f'Trajectory {labels[i]}')
            #ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], '.', color=color, markersize=3)
        # Plot target points
        if target_points is not None and target_labels is not None:
            for i in range(len(target_labels)):
                target_label = target_labels[i]
                color = label_to_color[target_label]
                ax.scatter(target_points[0,0,i], target_points[0,1,i], target_points[0,2,i], marker='D', color=color, s=200, alpha=1.0)#, label=f'Target {target_label}')
        #ax.set_xlabel('X')
        #ax.set_ylabel('Y')
        #ax.set_zlabel('Z')
        if ax_view is not None:
            ax.view_init(elev=ax_view[0], azim=ax_view[1])

    #plt.title(title)
    # Avoid duplicate labels in legend
    handles, labels_ = ax.get_legend_handles_labels()
    unique = dict(zip(labels_, handles))
    ax.legend(unique.values(), unique.keys())
    plt.tight_layout()
    if filepath is not None:
        plt.savefig(Path(str(filepath) + ".eps"), format='eps')
        plt.savefig(Path(str(filepath) + ".png"), format='png', dpi=300)
    plt.show()


def find_run_directory(reconstruction_model, run_id=None, saves_dir='saves/classification'):
    """
    Find the run directory for a trained model.
    
    Parameters
    ----------
    reconstruction_model : str
        Name of the reconstruction model used for training
    run_id : str, optional
        Specific run ID. If None, returns the most recent run.
    saves_dir : str or Path, optional
        Base directory where classification models are saved
    
    Returns
    -------
    Path
        Path to the run directory
    """
    saves_dir = Path(saves_dir)
    model_dir = saves_dir / reconstruction_model
    
    if not model_dir.exists():
        raise FileNotFoundError(
            f"No classification models found for reconstruction model: {reconstruction_model}\n"
            f"Expected directory: {model_dir}"
        )
    
    # Find the run directory
    if run_id is None:
        # Get the most recent run (timestamp directories start with a digit)
        run_dirs = [d for d in model_dir.iterdir() if d.is_dir() and d.name[0].isdigit()]
        if not run_dirs:
            raise ValueError(f"No training runs found in {model_dir}")
        run_dirs.sort(reverse=True)  # Most recent first
        run_path = run_dirs[0]
        print(f"Loading most recent run: {run_path.name}")
    else:
        run_path = model_dir / run_id
        if not run_path.exists():
            raise FileNotFoundError(f"Run directory not found: {run_path}")
        print(f"Loading run: {run_id}")
    
    return run_path


def load_model_from_config(reconstruction_model, run_id=None, saves_dir='saves/classification'):
    """
    Load a trained classification model using existing library functions.
    
    Uses TrainingState.load() and the autoencoder loading from setup().
    
    Parameters
    ----------
    reconstruction_model : str
        Name of the reconstruction model used for training
    run_id : str, optional
        Specific run ID (e.g., '20260209_143022'). If None, loads the most recent run.
    saves_dir : str or Path, optional
        Base directory where classification models are saved
    
    Returns
    -------
    tuple
        (trainingstate, config, autoencoder, hpara, run_path)
    
    Example
    -------
    ```python
    trainingstate, config, X_ae, hpara, run_path = load_model_from_config(
        'Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1'
    )
    ```
    """
    # Find the run directory
    run_path = find_run_directory(reconstruction_model, run_id, saves_dir)
    
    # Load configuration
    config_path = run_path / 'config.json'
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
        
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    print(f"Loaded configuration from: {config_path}")
    print(f"Model trained on: {config['metadata']['timestamp']}")
    print(f"Best validation accuracy: {config['performance']['best_val_accuracy']:.4f}")
    print(f"Best epoch: {config['performance']['best_epoch']}")
    
    # Load training state using library function
    trainingstate = TrainingState.load(run_path)
    print(f"Loaded training state from: {run_path}")
    
    # Load autoencoder using the same approach as setup()
    from polynode import util_lib, model_lib
    
    saves_root = Path(__file__).parent.parent.parent / "saves" / "reconstruction"
    autoenc_path = saves_root / config['reconstruction']['model_name']
    
    hpara = util_lib.read_report(autoenc_path / "config.txt")
    hpara["device"] = torch.device('cuda:' + str(hpara["gpu"]) if torch.cuda.is_available() else 'cpu')
    
    with hpara["device"]:
        X_ae = model_lib.CompVFReduced(
                    width=hpara["width"], 
                    dim_comp=hpara["dim_comp"], 
                    dim_free=hpara["dim_free"]
                )
        X_ae = X_ae.to(hpara["device"])
        X_ae.x1_free = hpara["x1_free"]
        X_ae.x1_free_buffer = hpara["x1_free_buffer"]
        if "X1_speed" in hpara:
            X_ae.X1_speed = hpara["X1_speed"]
        X_ae.load_state_dict(torch.load(autoenc_path / "checkpoint.pt", weights_only=True, map_location=hpara["device"]))
    
    print(f"Loaded autoencoder from: {autoenc_path}\n")
    
    return trainingstate, config, X_ae, hpara, run_path


def list_available_models(saves_dir='saves/classification'):
    """
    List all available trained classification models organized by reconstruction model.
    
    Parameters
    ----------
    saves_dir : str or Path, optional
        Base directory where classification models are saved
    
    Returns
    -------
    dict
        Dictionary mapping reconstruction model names to lists of run directories
    
    Example
    -------
    ```python
    models = list_available_models()
    for recon_model, runs in models.items():
        print(f"{recon_model}: {len(runs)} runs")
    ```
    """
    saves_dir = Path(saves_dir)
    
    if not saves_dir.exists():
        print(f"No classification models found at {saves_dir}")
        return {}
    
    models = {}
    for recon_model_dir in saves_dir.iterdir():
        if recon_model_dir.is_dir():
            # Look for timestamp directories (start with a digit)
            runs = [d.name for d in recon_model_dir.iterdir() 
                   if d.is_dir() and d.name[0].isdigit()]
            if runs:
                models[recon_model_dir.name] = sorted(runs, reverse=True)
    
    return models


def print_available_models(saves_dir='saves/classification'):
    """
    Print all available classification models in a readable format.
    
    Parameters
    ----------
    saves_dir : str or Path, optional
        Base directory where classification models are saved
    """
    models = list_available_models(saves_dir)
    
    if not models:
        print(f"No trained classification models found in {saves_dir}")
        return
    
    print("\n" + "="*70)
    print("AVAILABLE CLASSIFICATION MODELS")
    print("="*70)
    
    for recon_model, runs in models.items():
        print(f"\n{recon_model}:")
        for run in runs:
            run_path = Path(saves_dir) / recon_model / run
            config_path = run_path / 'config.json'
            
            if config_path.exists():
                with open(config_path, 'r') as f:
                    config = json.load(f)
                
                timestamp = config['metadata']['timestamp']
                best_acc = config['performance']['best_val_accuracy']
                best_epoch = config['performance']['best_epoch']
                vftype = config['model_architecture']['vf_type']
                method = config['ode_solver']['method']
                print(f"  - {run}")
                print(f"      Timestamp: {timestamp}, Best val acc: {best_acc:.4f} (epoch {best_epoch})")
                print(f"      VF type: {vftype}, ODE method: {method}")
            else:
                print(f"  - {run} (config not found)")
    
    print("="*70 + "\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Evaluate trained classification model',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
        Examples:
        # List all available models
        python -m polynode.classification.classification_evaluate --list_models
        
        # Evaluate most recent run (generates plots saved to model directory)
        python -m polynode.classification.classification_evaluate \\
            --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1
        
        # Evaluate specific run
        python -m polynode.classification.classification_evaluate \\
            --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1 \\
            --run_id 20260209_143022
        
        # Evaluate with interactive Dash app (opens at http://localhost:8052)
        python -m polynode.classification.classification_evaluate \\
            --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_1 \\
            --dash
        """
    )
    parser.add_argument('--reconstruction_model', type=str, default=None,
                       help='Name of the reconstruction model (subfolder in saves/classification/)')
    parser.add_argument('--run_id', type=str, default=None,
                       help='Specific run ID to load (e.g., 20260209_143022). If not specified, uses most recent.')
    parser.add_argument('--list_models', action='store_true',
                       help='List all available trained models and exit')
    parser.add_argument('--saves_dir', type=str, default='saves/classification',
                       help='Base directory where classification models are saved')
    parser.add_argument('--dash', action='store_true',
                       help='Open interactive Dash visualization app')
    parser.add_argument('--port', type=int, default=8052,
                       help='Port for Dash visualization app (only used with --dash)')
    parser.add_argument('--skip_plots', action='store_true',
                       help='Skip generating trajectory plots (only show stats)')
    
    args = parser.parse_args()
    
    # List models if requested
    if args.list_models:
        print_available_models(args.saves_dir)
    elif args.reconstruction_model:
        # Load model using library functions
        trainingstate, config, autoencoder, hpara, run_dir = load_model_from_config(
            args.reconstruction_model,
            args.run_id,
            args.saves_dir
        )
        
        print("\n" + "="*70)
        print("MODEL CONFIGURATION")
        print("="*70)
        print(json.dumps(config, indent=2))
        print("="*70 + "\n")
        
        print("Model loaded successfully!")
        print(f"Final Training Accuracy: {trainingstate.train_accuracy_list[-1]:.2%}")
        print(f"Final Validation Accuracy: {trainingstate.val_accuracy_list[-1]:.2%}")
        print(f"Final Training MSE Loss: {trainingstate.train_mse_loss_list[-1]:.4f}")
        print(f"Final Validation MSE Loss: {trainingstate.val_mse_loss_list[-1]:.4f}")
        
        # Run evaluation and generate plots
        if not args.skip_plots:
            print("\n" + "="*70)
            print("RUNNING EVALUATION")
            print("="*70)
            
            # Get dataloaders using the loaded hyperparameters
            train_loader, val_loader, t, t_latent = class_spiral_lib.get_dataloaders(hpara)
            
            # Recreate args object from config for compatibility with evaluation functions
            eval_args = argparse.Namespace(
                width=config['model_architecture']['width'],
                nlayer=config['model_architecture']['num_layers'],
                nfreqs=config['model_architecture']['num_freqs'],
                vftype=config['model_architecture']['vf_type'],
                interpolant=config['model_architecture']['interpolant'],
                epoch=config['training']['epochs'],
                bsize=config['training']['batch_size'],
                weight_decay=config['training']['weight_decay'],
                opt=config['training']['optimizer'],
                method=config['ode_solver']['method'],
                nt=config['ode_solver']['num_timesteps'],
                radprob=config['data']['use_radial_probs'],
                nlabels=config['data']['num_labels'],
                attractor_strength=config['data']['attractor_strength'],
                attractor_scale=config['data']['attractor_scale'],
                reconstruction_model=config['reconstruction']['model_name']
            )
            
            # Get classifier and evaluation functions
            X_class = class_spiral_lib.get_classification_vf(hpara, eval_args)
            X_class.load_state_dict(trainingstate.best_model_state)
            X_class.eval()
            
            loss_func, eval_func = class_spiral_lib.get_loss_and_eval(X_class, autoencoder, eval_args, hpara)
            
            # Evaluate on validation set
            x_val, y_val = next(iter(val_loader))
            with torch.no_grad():
                loss_func_output = loss_func(x_val, y_val, t_latent)
                val_loss = loss_func_output[0]
                val_output = loss_func_output[1]
                val_accuracy = loss_func_output[2]
                val_mse_loss = loss_func_output[3]
                val_labels = loss_func_output[4]
            
            print(f"Generating plots...")
            
            # Plot validation accuracy over epochs
            plt.figure()
            plt.plot(trainingstate.val_accuracy_list)
            plt.title("Validation Accuracy over Epochs")
            plt.xlabel("Epoch")
            plt.ylabel("Accuracy")
            plt.tight_layout()
            val_acc_plot = run_dir / "eval_val_accuracy"
            plt.savefig(str(val_acc_plot) + ".png", dpi=300)
            plt.savefig(str(val_acc_plot) + ".eps", format='eps')
            plt.show()
            print(f"✓ Saved: {val_acc_plot}.png and {val_acc_plot}.eps")
            
            # Plot validation trajectories (subsample to reduce clutter)
            skip_every_n = 24
            val_output_np = val_output[:, ::skip_every_n].detach().cpu().numpy()
            val_labels_np = val_labels[::skip_every_n].detach().cpu().numpy()
            target_points_np = class_spiral_lib.get_targets(hpara).cpu().numpy()
            
            val_plot_path = run_dir / "eval_val_trajectories"
            plot_trajectories_eval(
                val_output_np,
                labels=val_labels_np,
                title="Validation Trajectories",
                target_points=target_points_np,
                target_labels=torch.arange(config['data']['num_labels']).numpy(),
                ax_view=(30, 45),
                filepath=val_plot_path
            )
            print(f"✓ Saved: {val_plot_path}.png and {val_plot_path}.eps")
            
            # Plot loss curves
            fig_loss, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
            
            ax1.plot(trainingstate.train_mse_loss_list, label='Train')
            ax1.plot(trainingstate.val_mse_loss_list, label='Val')
            ax1.set_xlabel('Epoch')
            ax1.set_ylabel('MSE Loss')
            
            print(f"\nAll plots saved to: {run_dir}")
            
            # Optionally open dash app for interactive visualization
            if args.dash:
                print("\n" + "="*70)
                print("STARTING INTERACTIVE VISUALIZATION")
                print("="*70)
                
                # Need to evaluate on train set for dash
                x_train, y_train = next(iter(train_loader))
                with torch.no_grad():
                    train_output, train_labels, train_labelpoints = eval_func(x_train, y_train, t_latent)
                
                # Start dash app
                import matplotlib
                matplotlib.use('Agg')  # Switch to non-interactive backend for dash compatibility
                
                model_name = f"{args.reconstruction_model}_{run_dir.name}"
                shared_data, dash_proc = pp.start_dash_app_multiprocessing(
                    kwargs={"port": args.port, "title": model_name}
                )
                
                # Generate plots
                class_spiral_lib.dash_plotting(
                    shared_data, 
                    hpara, 
                    trainingstate,
                    (train_labels, val_labels),
                    (train_output, val_output)
                )
                
                print(f"\n✓ Visualization server running at http://localhost:{args.port}")
                print("Press Ctrl+C to stop the server...")
                
                try:
                    # Keep server running
                    while not class_spiral_lib.should_shutdown():
                        time.sleep(1)
                except KeyboardInterrupt:
                    print("\nShutting down visualization server...")
                finally:
                    if dash_proc:
                        dash_proc.terminate()
                        dash_proc.join()
                    print("Server stopped.")
        
    else:
        parser.print_help()


