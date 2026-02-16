"""  
Utilities for 3D spiral classification tasks.

This module contains specialized functions and classes for training neural ODE classifiers
on 3D spiral trajectories.

Key components:
    - TrainingState: Class for managing and persisting training state with JSON configs
    - setup(): Configure training environment and load reconstruction models
    - generate_data(): Generate spiral trajectory data for classification
    - get_classification_vf(): Create classification vector fields
    - get_loss_and_eval(): Define loss functions and evaluation metrics
    - Label positioning functions: xlabelfunc, ylabelfunc, zlabelfunc for target placement

Example:
    ```python
    # Setup training environment
    shared_data, dash_proc, args, hpara, X_ae, model_name = setup()
    
    # Get data loaders
    train_loader, val_loader, t, t_latent = get_dataloaders(hpara)
    
    # Create classification model
    X_class = get_classification_vf(hpara, args)
    
    # Get loss and evaluation functions
    loss_func, eval_func = get_loss_and_eval(X_class, X_ae, args, hpara)
    ```
"""

import torch
from polynode import data_lib
from polynode import util_lib
from polynode import model_lib

import pickle
import json
import time
import math
from datetime import datetime
from pathlib import Path

from polynode.classification import classification_lib as class_lib

import polynode.plotly_plot as pp

class TrainingState:
    """
    Manages training state and saves models with configuration for classification experiments.
    
    This class tracks training and validation metrics throughout the training process,
    stores the best model state in memory, and saves everything to disk including
    a JSON configuration file for easy loading and reproducibility.
    
    Attributes
    ----------
    args : argparse.Namespace
        Command-line arguments containing training hyperparameters
    hpara : dict
        Hyperparameters from the reconstruction model configuration
    best_model_state : dict or None
        State dictionary of the best model (by validation accuracy)
    best_model_epoch : int
        Epoch number when best model was saved
    train_mse_loss_list : list
        MSE loss values for each training epoch
    train_accuracy_list : list
        Accuracy values for each training epoch
    val_mse_loss_list : list
        MSE loss values for each validation epoch
    val_accuracy_list : list
        Accuracy values for each validation epoch
    """
    def __init__(self, args, hpara):
        self.args = args
        self.hpara = hpara

        self.best_model_state = None  # Store the best model state in memory
        self.best_model_epoch = -1

        self.train_mse_loss_list = []
        self.train_accuracy_list = []

        self.val_mse_loss_list = []
        self.val_accuracy_list = []

    def save(self, dirpath):
        """
        Save training state, model, and configuration to organized directory structure.
        
        Creates a directory structure organized by reconstruction model and timestamp,
        saves the best model state dict, training statistics pickle, and a JSON
        configuration file with all hyperparameters and performance metrics.
        
        Parameters
        ----------
        dirpath : Path or str
            Directory path where files will be saved. This should be the run directory
            (e.g., saves/classification/{reconstruction_model}/{timestamp})
        
        Saves
        -----
        - best_model.pt : Best model state dictionary
        - training_stats.pkl : Pickled TrainingState object
        - config.json : JSON file with all configuration and performance metrics
        - training_history.json : JSON file with loss/accuracy history
        """
        dirpath = Path(dirpath)
        dirpath.mkdir(parents=True, exist_ok=True)
        
        # Save model state dict
        state_dict_path = dirpath / "best_model.pt"
        if self.best_model_state is not None:
            torch.save(self.best_model_state, state_dict_path)
        
        # Save training stats pickle (for backward compatibility)
        self.best_model_state = None  # Don't pickle the state dict
        with open(dirpath / "training_stats.pkl", 'wb') as f:
            pickle.dump(self, f)
        
        # Create and save JSON configuration
        config = {
            'model_architecture': {
                'width': self.args.width,
                'num_layers': self.args.nlayer,
                'num_freqs': self.args.nfreqs,
                'vf_type': self.args.vftype,
                'interpolant': self.args.interpolant,
                'input_dim': 3,  # latent dim for spiral classification
                'output_dim': self.args.nlabels,
            },
            'training': {
                'epochs': self.args.epoch,
                'batch_size': self.args.bsize,
                'weight_decay': self.args.weight_decay,
                'optimizer': self.args.opt,
            },
            'ode_solver': {
                'method': self.args.method,
                'num_timesteps': self.args.nt,
            },
            'data': {
                'use_radial_probs': self.args.radprob,
                'num_labels': self.args.nlabels,
                'attractor_strength': self.args.attractor_strength,
                'attractor_scale': self.args.attractor_scale,
            },
            'reconstruction': {
                'model_name': self.args.reconstruction_model,
                'latent_dim': 3,
            },
            'metadata': {
                'timestamp': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'save_directory': str(dirpath),
                'gpu_name': torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu',
                'pytorch_version': torch.__version__,
            },
            'performance': {
                'best_epoch': self.best_model_epoch,
                'best_val_accuracy': float(max(self.val_accuracy_list)) if self.val_accuracy_list else 0.0,
                'final_train_mse': float(self.train_mse_loss_list[-1]) if self.train_mse_loss_list else None,
                'final_train_accuracy': float(self.train_accuracy_list[-1]) if self.train_accuracy_list else None,
                'final_val_mse': float(self.val_mse_loss_list[-1]) if self.val_mse_loss_list else None,
                'final_val_accuracy': float(self.val_accuracy_list[-1]) if self.val_accuracy_list else None,
            }
        }
        
        # Save configuration as JSON
        config_path = dirpath / 'config.json'
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=4)
        
        # Save training history as separate JSON
        history = {
            'train_mse_loss': [float(x) for x in self.train_mse_loss_list],
            'train_accuracy': [float(x) for x in self.train_accuracy_list],
            'val_mse_loss': [float(x) for x in self.val_mse_loss_list],
            'val_accuracy': [float(x) for x in self.val_accuracy_list],
            'epochs': list(range(len(self.train_mse_loss_list))),
        }
        history_path = dirpath / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=4)
        
        print(f"\nModel and configuration saved to: {dirpath}")
        print(f"  - Best model: {state_dict_path}")
        print(f"  - Configuration: {config_path}")
        print(f"  - Training history: {history_path}")

    @staticmethod
    def load(dirpath):
        """
        Load training state from saved files.
        
        Handles backward compatibility with old pickle files that reference
        the old module name 'classification_spiral_3D_lib'.
        
        Parameters
        ----------
        dirpath : Path or str
            Directory path where files are saved
        
        Returns
        -------
        TrainingState
            Loaded training state with best model attached
        """
        dirpath = Path(dirpath)
        
        # Load model state dict
        state_dict_path = dirpath / "best_model.pt"
        if state_dict_path.exists():
            best_model_state = torch.load(state_dict_path)
        else:
            best_model_state = None

        # Load training stats pickle with module name remapping for backward compatibility
        import sys
        import polynode.classification.classification_spiral_lib
        
        # Temporarily alias old module name to current module for pickle compatibility
        sys.modules['classification_spiral_3D_lib'] = polynode.classification.classification_spiral_lib
        
        try:
            with open(dirpath / "training_stats.pkl", 'rb') as f:
                trainingstate = pickle.load(f)
            trainingstate.best_model_state = best_model_state
        finally:
            # Clean up the temporary alias
            if 'classification_spiral_3D_lib' in sys.modules:
                del sys.modules['classification_spiral_3D_lib']
        
        return trainingstate

def setup(radprob=True):
    """
    Setup training environment, parse arguments, and load reconstruction model.
    
    Configures the training environment by parsing command-line arguments,
    loading the specified reconstruction autoencoder model, and initializing
    the visualization dashboard.
    
    Parameters
    ----------
    radprob : bool, optional
        Default value for radius-based classification. If True, classifies based
        on radial distance; if False, uses angular/quadrant classification.
        Default is True.
    
    Returns
    -------
    tuple
        (shared_data, dash_proc, args, hpara, X_ae, model_name)
        - shared_data : dict
            Shared dictionary for dashboard visualization
        - dash_proc : Process
            Dashboard process for live plotting
        - args : argparse.Namespace
            Parsed command-line arguments
        - hpara : dict
            Hyperparameters from reconstruction model
        - X_ae : nn.Module
            Loaded autoencoder model
        - model_name : str
            Descriptive name for this training run
    
    Example
    -------
    ```python
    # Setup with default parameters
    shared_data, dash_proc, args, hpara, X_ae, model_name = setup()
    
    # Setup with custom reconstruction model
    # python classification.py --reconstruction_model Spiral_nd_CompVFReduced_poly_structure_4_2_speed_2
    ```
    """
    import argparse

    parser = argparse.ArgumentParser(description="Run classification training with custom settings.")
    parser.add_argument("--port", type=int, default=8051, help="Port to run the Dash app on.")
    parser.add_argument("--method", default="euler", help="ODE method to use.")
    parser.add_argument("--nt", type=int, default=150, help="Number of time steps.")
    parser.add_argument("--width", type=int, default=512, help="Width of Classification VF")
    parser.add_argument("--nlayer", type=int, default=2, help="Number of layers in Classification VF")
    parser.add_argument("--nfreqs", type=int, default=0, help="Number of terms in TimeShallow VF")
    parser.add_argument("--epoch", type=int, default=5, help="Number of epochs for training")
    parser.add_argument("--weight_decay", type=float, default=1e-6, help="Weight decay for optimizer")
    parser.add_argument("--opt", type=str, default="adam", help="Optimizer to use (adam, sgd, etc.)")
    parser.add_argument("--bsize", type=int, default=32, help="Batch size for training")
    parser.add_argument("--radprob", type=bool, default=radprob, help="Use radius based classification")
    parser.add_argument("--nlabels", type=int, default=3, help="Number of labels for classification")
    parser.add_argument("--attractor_strength", type=float, default=50.0, help="Attractor strength for classification VF")
    parser.add_argument("--attractor_scale", type=float, default=4.0, help="Attractor scale for classification VF")
    parser.add_argument("--interpolant", type=str, default="polynomial", help="Type of interpolant to use in TimeShallow layers (fourier, polynomial)")
    parser.add_argument("--vftype", type=str, default="classic", help="Type of vector field (classic, timeshallow)")
    parser.add_argument("--reconstruction_model", type=str,
                        default="SpiralND_CompVFReduced_poly_structure_4_2_speed_5.0",
                        help="Name of the reconstruction model to use (subfolder in saves/reconstruction/)")


    args = parser.parse_args()

    # Create simplified model name for dashboard title
    model_name = f"spiral_{args.vftype}_method:{args.method}_nt:{args.nt}"
    if args.vftype == "timeshallow":
        model_name += f"_nfreqs:{args.nfreqs}"
    model_name += f"_epoch:{args.epoch}"

    print(f"Model name: {model_name}")
    print(f"Reconstruction model: {args.reconstruction_model}")

    shared_data, dash_proc = pp.start_dash_app_multiprocessing(
        kwargs={"port": args.port, "title": model_name}
    )

    # Load reconstruction model from saves directory
    saves_root = Path(__file__).parent.parent.parent / "saves" / "reconstruction"
    autoenc_path = saves_root / args.reconstruction_model
    
    if not autoenc_path.exists():
        raise FileNotFoundError(
            f"Reconstruction model not found at: {autoenc_path}\n"
            f"Available models in saves/reconstruction/: "
            f"{[d.name for d in saves_root.iterdir() if d.is_dir()]}"
        )

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
        X_ae.load_state_dict(torch.load(autoenc_path / "checkpoint.pt", weights_only=True, map_location=torch.device(hpara["device"])))

    return shared_data, dash_proc, args, hpara, X_ae, model_name


def generate_data(hpara, arg_index):
    """
    Generate synthetic spiral trajectory data for classification.
    
    Creates training and validation datasets of spiral trajectories with specified
    properties including speed, radius, and angular characteristics.
    
    Parameters
    ----------
    hpara : dict
        Hyperparameters dictionary containing device, ODE method, etc.
    arg_index : int
        Index to select spiral speed from predefined list
    
    Returns
    -------
    tuple
        (t, dataset_train, dataset_val)
        - t : torch.Tensor
            Time points for trajectory integration
        - dataset_train : torch.utils.data.TensorDataset
            Training dataset
        - dataset_val : torch.utils.data.TensorDataset
            Validation dataset
    """
    # data setup
    # time        
    t=data_lib.get_time(hpara)
    
    # transformation       
    #shift
    x1_shift = torch.tensor((15., 0.,  0., 0.))
    shift= data_lib.Shift(x1_shift)
    factors = torch.tensor([1., 1.5, 1., 1., 2.3])
    scale = data_lib.Scale(factors)
    trafo= shift # data_lib.Composition_Trafo(shift, scale)  # 
    
    # parametrisation
    # spiral
    speed_list = [1, 2, 3, 4, 5] #    [0.5, 1.5, 2.5, 3.5, 4.5]  #  [0.5*i for i in range(1,11)] # 
    x0= torch.tensor([-7.,6.,0., 0.]) 
    speed_radius= 0.5
    r_min=1
    angel_factor= 2
    phi_0 = 0 #-torch.pi/2
    speed_angle = speed_list[int(arg_index)]
    e1 = torch.tensor([0., 1., 0.,  0.])
    e2 = torch.tensor([0., 0., 0.,  1.])
    e2 = e2/torch.linalg.vector_norm(e2)
    basis = (e2, e1 )
    phi_para = data_lib.SpiralND(x0, speed_radius,speed_angle=speed_angle , phi_0= phi_0, basis= basis, r_min=r_min)
    bounds = torch.tensor( ((0.0, angel_factor*torch.pi),) )
    
    # calc internal length to adjust the number of samples
    dist = data_lib.DistanceSpiral(phi_para)
    y_temp= phi_para(bounds[0])
    length_intrinsic = (dist(y_temp)[1] - dist(y_temp)[0]).item()
    
    sample_method_train= "1d_uniform_intrinsic"
    sample_size_train =   torch.tensor([int(round( min(20*length_intrinsic, 5000), -2))])
    sample_method_val=  "random"
    sample_size_val = torch.tensor([int(sample_size_train[0]*0.3)]) # [333]

    data_train = data_lib.DatasetSynth(phi_para, bounds, trafo, note="training", device=hpara["device"])
    dataset_train = data_train.sample(sample_size_train, sample_method_train, dist=dist) 

    data_val = data_lib.DatasetSynth(phi_para, bounds, trafo, note="validation", device=hpara["device"])
    dataset_val = data_val.sample(sample_size_val, sample_method_val)

    return (t, dataset_train, dataset_val)

def get_dataloaders(hpara):
    """
    Create PyTorch data loaders for training and validation.
    
    Parameters
    ----------
    hpara : dict
        Hyperparameters dictionary
    
    Returns
    -------
    tuple
        (train_loader, val_loader, t, t_latent)
        - train_loader : DataLoader
            Training data loader
        - val_loader : DataLoader
            Validation data loader
        - t : torch.Tensor
            Full time range
        - t_latent : torch.Tensor
            Time range for latent encoding (first half)
    """
    (t, dtrain, dval) = generate_data(hpara, 4)
    t = t.to(hpara["device"])
    train_loader = torch.utils.data.DataLoader(dtrain, batch_size=32, shuffle=True, drop_last=True)
    val_loader = torch.utils.data.DataLoader(dval, batch_size=dval.tensors[0].shape[0], shuffle=True, drop_last=False)
    t_latent = t[0:int(len(t)/2)]
    return train_loader, val_loader, t, t_latent

# def xlabelfunc(label):
#     pos = torch.empty(label.shape, device=label.device, dtype=torch.float32)
#     pos[label == 0] = 0.485 + 1.0
#     pos[(label == 1) | (label == 2)] = 0.485 - 1/math.sqrt(2)
#     if label.max() > 2 or label.min() < 0:
#         raise ValueError(f"Unknown label: {label}")
#     return pos

def xlabelfunc(label):
    """
    Map classification labels to x-coordinates of target points in latent space.
    
    Parameters
    ----------
    label : torch.Tensor
        Class labels (0, 1, or 2)
    
    Returns
    -------
    torch.Tensor
        X-coordinates for target points
    """
    pos = torch.empty(label.shape, device=label.device, dtype=torch.float32)
    pos[label == 0] = 4.0
    pos[label == 1] = 5.0
    pos[label == 2] = 4.0
    #pos[(label == 1) | (label == 2)] = 5.0
    if label.max() > 2 or label.min() < 0:
        raise ValueError(f"Unknown label: {label}")
    return pos

# def ylabelfunc(label):
#     pos = torch.empty(label.shape, device=label.device, dtype=torch.float32)
#     pos[label == 0] = 0.0
#     pos[label == 1] = 1.0/math.sqrt(2)
#     pos[label == 2] = -1.0/math.sqrt(2)
#     if label.max() > 2 or label.min() < 0:
#         raise ValueError(f"Unknown label: {label}")
#     return pos

def ylabelfunc(label):
    """
    Map classification labels to y-coordinates of target points in latent space.
    
    Parameters
    ----------
    label : torch.Tensor
        Class labels (0, 1, or 2)
    
    Returns
    -------
    torch.Tensor
        Y-coordinates for target points
    """
    pos = torch.empty(label.shape, device=label.device, dtype=torch.float32)
    pos[label == 0] = -3.0
    pos[label == 1] = 0.0
    pos[label == 2] = 3.0
    if label.max() > 2 or label.min() < 0:
        raise ValueError(f"Unknown label: {label}")
    return pos

# def zlabelfunc(label):
#     return torch.zeros_like(label, device=label.device, dtype=torch.float32)

def zlabelfunc(label):
    """
    Map classification labels to z-coordinates of target points in latent space.
    
    Parameters
    ----------
    label : torch.Tensor
        Class labels (0, 1, or 2)
    
    Returns
    -------
    torch.Tensor
        Z-coordinates for target points (currently all zeros)
    """
    pos = torch.empty(label.shape, device=label.device, dtype=torch.float32)
    pos[label == 0] = 0.0
    pos[label == 1] = 0.0
    pos[label == 2] = 0.0
    if label.max() > 2 or label.min() < 0:
        raise ValueError(f"Unknown label: {label}")
    return pos

def get_targets(hpara):
    """
    Generate target points in latent space for each classification label.
    
    Parameters
    ----------
    hpara : dict
        Hyperparameters dictionary containing device
    
    Returns
    -------
    torch.Tensor
        Target points tensor of shape (1, 3, num_labels) containing (x, y, z)
        coordinates for each label's target in latent space
    """
    targets = torch.tensor(
        [[
            xlabelfunc(torch.scalar_tensor(l)).item(),
            ylabelfunc(torch.scalar_tensor(l)).item(),
            zlabelfunc(torch.scalar_tensor(l)).item()
        ] for l in range(3)]
    ).transpose(0,1)[None,...].contiguous().to(hpara["device"])
    #target_points = targets[0,:,:].transpose(0,1)
    return targets


def get_classification_vf(hpara, args):
    """
    Create classification vector field model.
    
    Parameters
    ----------
    hpara : dict
        Hyperparameters dictionary
    args : argparse.Namespace
        Command-line arguments
    
    Returns
    -------
    nn.Module
        Classification vector field model (ClassificationVF or ClassificationVF_TimeShallow)
    """
    with hpara["device"]:
        if args.vftype == "classic":
            X_class = class_lib.ClassificationVF(
                            dim=3, 
                            width=args.width, 
                            nlayers=args.nlayer, 
                            targets= get_targets(hpara), 
                            attractor_scale=args.attractor_scale,
                            attractor_strength=args.attractor_strength
                        )
        elif args.vftype == "timeshallow":
            #X_class = ClassificationVF(dim=dim, width=args.width, nlayers=args.nlayer, targets=targets, attractor_strength=1e2)
            X_class = class_lib.ClassificationVFTimeShallow(
                            dim=3, 
                            device=hpara["device"],
                            width=args.width,
                            nlayers=args.nlayer,
                            nfreqs=args.nfreqs,
                            interpolant=args.interpolant,
                            targets=get_targets(hpara), 
                            attractor_scale=args.attractor_scale,
                            attractor_strength=args.attractor_strength,
                        )
    return X_class

def get_loss_and_eval(X_class, X_ae, args, hpara):
    """
    Create loss and evaluation functions for classification training.
    
    Parameters
    ----------
    X_class : nn.Module
        Classification vector field model
    X_ae : nn.Module
        Autoencoder model
    args : argparse.Namespace
        Command-line arguments
    hpara : dict
        Hyperparameters dictionary
    
    Returns
    -------
    tuple
        (classifier_loss, evaluate)
        - classifier_loss : function
            Loss function for training
        - evaluate : function
            Evaluation function for inference
    """
    x0 = hpara["x0"]
    radius_classifier = args.radprob
    quadrant_classifier = not radius_classifier
    max_radius = 1.0 + math.pi + 1e-5
    dim=3
    nlabels=3
    t_train = torch.linspace(
                    0.0,1.0,
                    args.nt, 
                    device=hpara["device"], 
                    dtype=torch.float32
                )
    targets = get_targets(hpara)

    if hpara["use_ode_adjoint"]:
        from torchdiffeq import odeint_adjoint as odeint
    else:
        from torchdiffeq import odeint

    def evaluate(x, y, t_latent):
        with torch.no_grad():
            if quadrant_classifier:
                angle = torch.atan2(x[:,1] - x0[1], x[:,3] - x0[3])
                angle += (angle < 0) * (2 * torch.pi)
                labels = torch.floor(nlabels * angle / (2 * torch.pi)).long()
            elif radius_classifier:
                r = torch.sqrt(torch.square((x[:,1] - x0[1])) + torch.square(x[:,3] - x0[3]))
                labels = torch.minimum((1e-6 + r*nlabels//max_radius), torch.scalar_tensor(nlabels-1)).long()
            else:
                raise ValueError("Either quadrant_classifier or radius_classifier must be True.")

            y_encode_trajectory = odeint(X_ae, x, t_latent, method=hpara["ode_method"])

        classifier_start = y_encode_trajectory[-1,:,:]
        classifier_start[:,1] = 0.0
        classifier_start[:,2] = 0.0
        if dim==3:
            classifier_start = classifier_start[:, [0,2,3]]

        output = odeint(X_class, classifier_start, 
                t_train,
                method=args.method)
        
        labelpoints = torch.zeros_like(classifier_start)
        labelpoints[:,0] = xlabelfunc(labels)
        labelpoints[:,1] = ylabelfunc(labels)
        labelpoints[:,2] = zlabelfunc(labels)
        
        return output, labels, labelpoints


    def classifier_loss(x, y, t_latent):

        output, labels, labelpoints = evaluate(x, y, t_latent)

        distance = torch.linalg.norm(output[-1][...,None] - targets, dim=1)

        mse_loss = torch.mean(torch.square(output[-1] - labelpoints).sum(dim=1))

        loss = mse_loss

        accuracy = (labels == torch.argmin(distance, dim=1)).float().mean().item()

        return loss, output, accuracy, mse_loss, labels

    return classifier_loss, evaluate

def get_optimizer(X_class, args):
    """
    Create optimizer for classification model.
    
    Parameters
    ----------
    X_class : nn.Module
        Classification model
    args : argparse.Namespace
        Command-line arguments
    
    Returns
    -------
    torch.optim.Optimizer
        Configured optimizer (Adam or SGD)
    """
    if args.opt == "adam":
        optimizer = torch.optim.Adam(
                        X_class.parameters(), 
                        lr=0.0005, 
                        weight_decay=args.weight_decay
                        )
    elif args.opt == "sgd":
        optimizer = torch.optim.SGD(
                        X_class.parameters(), 
                        lr=0.00002, 
                        momentum=0.9, 
                        weight_decay=args.weight_decay
                    )
    else:
        raise ValueError(f"Unknown optimizer: {args.opt}")
    return optimizer


def get_lr_scheduler(optimizer):
    """
    Create learning rate scheduler.
    
    Parameters
    ----------
    optimizer : torch.optim.Optimizer
        Optimizer to schedule
    
    Returns
    -------
    torch.optim.lr_scheduler.ReduceLROnPlateau
        Learning rate scheduler that reduces LR on plateau
    """
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                        optimizer, 
                        mode='min', 
                        factor=0.8, 
                        patience=20
                    )
    return lr_scheduler


def dash_plotting(shared_data, hpara, training_stats, labels, output):
    nlabels = 3
    target_points = get_targets(hpara)[0,:,:].transpose(0,1).detach()

    shared_data['train_trajectories'] = pp.plot_trajectories(output[0].detach(), 
                    labels=labels[0],
                    title="Train Output Trajectories",
                    target_points=target_points,
                    target_labels=torch.tensor([ilabel for ilabel in range(nlabels)]))
    
    shared_data['val_trajectories'] = pp.plot_trajectories(output[1].detach(), 
                    labels=labels[1],
                    title="Validation Output Trajectories",
                    target_points=target_points,
                    target_labels=torch.tensor([ilabel for ilabel in range(nlabels)]))

    shared_data['train_mse_loss_plot'] = pp.plot_loss(
        torch.tensor(training_stats.train_mse_loss_list), 
        title="Train MSE Loss", log_scale=True)
    shared_data['train_accuracy_plot'] = pp.plot_loss(
        torch.tensor(training_stats.train_accuracy_list), 
        title="Train Accuracy", log_scale=False)

    shared_data['val_mse_loss_plot'] = pp.plot_loss(
        torch.tensor(training_stats.val_mse_loss_list), 
        title="Val MSE Loss", log_scale=True)
    shared_data['val_accuracy_plot'] = pp.plot_loss(
        torch.tensor(training_stats.val_accuracy_list), 
        title="Val Accuracy", log_scale=False)

    shared_data['timestamp'] = time.time()

def should_shutdown():
    return pp.shutdown_flag.value

def print_stats(epoch, lr, saving_model, training_stats):
    # optimizer.param_groups[0]['lr']
    print(f"Epoch {epoch+1}, "
          f"TrainAcc: {training_stats.train_accuracy_list[-1]:.2f}, "
          f"TrainMSE: {training_stats.train_mse_loss_list[-1]:.2f}"
        )
    print(f"Epoch {epoch+1}, "
          f"ValAcc: {training_stats.val_accuracy_list[-1]:.2f}, "
          f"ValMSE: {training_stats.val_mse_loss_list[-1]:.2f}"
          )
    print(f"Epoch {epoch+1}, "
          f"Learning Rate: {lr:.2e}, "
          f"saving model: {saving_model}"
        )
    
