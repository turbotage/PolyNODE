"""
Neural network architectures for classification on dynamical systems.

This module provides neural ODE-based classification architectures that integrate
with pretrained autoencoders. The models learn to map encoded trajectories to
target points in latent space, with each target corresponding to a class label.

Key components:
    - ClassificationVF: Time-dependent vector field for classification
    - ClassificationVFTimeShallow: Time-shallow variant with frequency-based interpolation
    - TimeTransformation: Fourier/polynomial time-dependent transformations

Example:
    ```python
    from polynode.classification.classification_lib import ClassificationVF
    
    # Create classifier with target attractors
    targets = torch.tensor([[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]])
    classifier = ClassificationVF(
        dim=3,
        width=512,
        nlayers=2,
        targets=targets,
        attractor_strength=50.0
    )
    
    # Use in ODE integration
    from torchdiffeq import odeint
    t = torch.linspace(0, 1, 100)
    trajectory = odeint(classifier, x0, t)
    ```
"""

import torch
import torch.nn as nn


class ClassificationVF(nn.Module):
    """
    Time-dependent classification vector field with attractor dynamics.
    
    This neural ODE vector field learns to map initial points to target locations
    in latent space, where each target corresponds to a class. The field combines
    a learned neural network component with explicit attractor terms that pull
    trajectories toward their target classes.
    
    Attributes
    ----------
    dim : int
        Dimensionality of the latent space
    width : int
        Width of hidden layers
    nlayers : int
        Number of hidden layers
    targets : torch.Tensor or None
        Target points in latent space, shape (1, dim, num_classes)
    attractor_scale : float  
        Scale parameter for attractor decay
    attractor_strength : float
        Strength of attractor force 
    expscale : float
        Computed exponential scale based on target separation
    input_layer : nn.Linear
        Input projection layer (dim+1 -> width)
    hidden_layers : nn.ModuleList
        List of hidden layers with skip connections
    output_layer : nn.Linear
        Output projection layer (width -> dim)
    activ : nn.Module
        Activation function (ReLU)
    
    Example
    -------
    ```python
    # Create 3D classifier with 3 target classes
    targets = torch.tensor([[[4.0, 5.0, 4.0], [-3.0, 0.0, 3.0], [0.0, 0.0, 0.0]]])
    model = ClassificationVF(dim=3, width=512, nlayers=2, targets=targets)
    
    # Forward pass at time t
    t = torch.tensor(0.5)
    x = torch.randn(32, 3)  # Batch of 32 points
    dx_dt = model(t, x)
    ```
    """
    def __init__(self, dim, width=128, nlayers=4, targets=None, attractor_scale=2.0, attractor_strength=20.0):
        super().__init__()
        self.dim = dim
        self.width = width
        self.nlayers = nlayers
        self.targets = targets
        self.attractor_scale = attractor_scale
        self.attractor_strength = attractor_strength

        self.input_layer = nn.Linear(self.dim+1, self.width)
        self.hidden_layers = nn.ModuleList(
            [nn.Linear(self.width, self.width) for _ in range(self.nlayers)]
        )
        self.output_layer = nn.Linear(self.width, self.dim)

        self.activ = nn.ReLU() #nn.Tanh()

        self.expscale = 3.0
        if self.targets is not None:
            # Find the shortest distance between any two targets
            tt = self.targets[0,...].transpose(0,1)
            distance_pairs = torch.cdist(tt, tt)
            distance_pairs.fill_diagonal_( float('inf') )
            min_dist = distance_pairs.min()
            # We wan't the exp to decay at maximum 1/4'th of the distance between targets
            min_dist = min_dist / 4.0
            self.expscale = self.attractor_scale / torch.square(min_dist)  

        for m in self.hidden_layers:
            nn.init.xavier_uniform_(m.weight, gain=2.0)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.xavier_uniform_(self.output_layer.weight)

    def forward(self, t, x):
        """
        Compute the time derivative dx/dt at time t and state x.
        
        Parameters
        ----------
        t : torch.Tensor
            Scalar time value
        x : torch.Tensor
            State tensor of shape (batch_size, dim)
        
        """
        input = torch.cat([x, t.repeat(x.shape[0]).unsqueeze(1)], axis=1)
        h = self.activ(self.input_layer(input))
        # We use skip connections for better training stability
        for layer in self.hidden_layers:
            h = h + self.activ(layer(h))

        h = self.output_layer(h)

        # Add target attractors
        if self.targets is not None:
            vdir = self.targets - x[...,None] # Pointing from x to targets
            vdir = vdir / (torch.linalg.norm(vdir, dim=1, keepdim=True) + 1e-7)
            d = torch.linalg.norm(vdir, dim=1)
            attractor_mag = self.attractor_strength * torch.exp(-self.expscale * torch.square(d))
            vdir = vdir * attractor_mag.unsqueeze(1)
            h = h + vdir.sum(dim=2)

        return h

class TimeTransformation(nn.Module):
    def __init__(self, device, dim_in, dim_out, as_vector, n_freqs, df = 2*torch.pi, interpolant='fourier'):
        with device:
            super().__init__()
            self.dim_in = dim_in
            self.dim_out = dim_out
            self.n_freqs = n_freqs
            self.as_vector = as_vector
            self.interpolant = interpolant

            if self.interpolant == 'fourier':
                if dim_in == dim_out and as_vector:
                    self.coeffs_cos0 = nn.Parameter(torch.empty((dim_in,)))
                    #nn.init.xavier_uniform_(self.coeffs_cos0)
                    nn.init.zeros_(self.coeffs_cos0) # we can't use xavier here since the input is 1 dimensional
                    self.coeffs_cos = nn.ParameterList([nn.Parameter(torch.zeros((dim_in,))) for _ in range(n_freqs)])
                    self.coeffs_sin = nn.ParameterList([nn.Parameter(torch.zeros((dim_in,))) for _ in range(n_freqs)])
                    self.freqs  = nn.ParameterList([nn.Parameter(df * (i+1) * torch.ones((dim_in,))) for i in range(n_freqs)])
                else:
                    self.coeffs_cos0 = nn.Parameter(torch.empty((dim_out, dim_in)))
                    nn.init.xavier_uniform_(self.coeffs_cos0)
                    #nn.init.normal_(self.coeffs_cos0, mean=0.0, std=1e-3) # we can't use xavier here since the input is 1 dimensional
                    self.coeffs_cos = nn.ParameterList([nn.Parameter(torch.zeros((dim_out, dim_in))) for _ in range(n_freqs)])
                    self.coeffs_sin = nn.ParameterList([nn.Parameter(torch.zeros((dim_out, dim_in))) for _ in range(n_freqs)])
                    self.freqs  = nn.ParameterList([nn.Parameter(df * (i+1) * torch.ones((dim_out, dim_in))) for i in range(n_freqs)])
            elif self.interpolant == 'polynomial':
                if dim_in == dim_out and as_vector:
                    self.coeffs = nn.ParameterList([nn.Parameter(torch.empty((dim_in,))) for _ in range(n_freqs+1)])
                    for i in range(n_freqs+1):
                        nn.init.zeros_(self.coeffs[i])
                else:
                    self.coeffs = nn.ParameterList([nn.Parameter(torch.empty((dim_out, dim_in))) for _ in range(n_freqs+1)])
                    for i in range(n_freqs+1):
                        nn.init.xavier_uniform_(self.coeffs[i], gain=1.0/(2**i))
            else:
                raise ValueError(f"Unknown interpolant: {self.interpolant}")


    def forward(self, t):
        if self.interpolant == 'fourier':
            out = self.coeffs_cos0.unsqueeze(0)
            for i in range(self.n_freqs):
                out = out + self.coeffs_cos[i].unsqueeze(0) * torch.cos(self.freqs[i].unsqueeze(0) * t)
                out = out + self.coeffs_sin[i].unsqueeze(0) * torch.sin(self.freqs[i].unsqueeze(0) * t)
            return out
        elif self.interpolant == 'polynomial':
            out = self.coeffs[0].unsqueeze(0)
            tnew = torch.ones_like(t)
            for i in range(0, self.n_freqs):
                out = out + self.coeffs[i+1].unsqueeze(0) * tnew
                tnew = tnew * t
            return out

# The transformation W(t)\sigma(V(t)x + b(t)) where the time dependence is given by a Fourier series or a Polynomial
class TimeShallowLayer(nn.Module):
    def __init__(self, device,
                 dim_in, dim_mid, dim_out, 
                 w_freqs=0, v_freqs=0, b_freqs=0,
                 w_freeze=False, v_freeze=False, b_freeze=False,
                 w_interpolant='fourier', v_interpolant='fourier', b_interpolant='fourier'):
        with device:
            super().__init__()
            self.device = device
            self.dim_in = dim_in
            self.dim_mid = dim_mid
            self.dim_out = dim_out
            self.w_freqs = w_freqs
            self.v_freqs = v_freqs
            self.b_freqs = b_freqs
            self.w_freeze = w_freeze
            self.v_freeze = v_freeze
            self.b_freeze = b_freeze
            self.w_interpolant = w_interpolant
            self.v_interpolant = v_interpolant
            self.b_interpolant = b_interpolant
            
            if self.v_freeze and self.dim_in != self.dim_mid:
                raise ValueError("Cannot freeze V(t) when dim_in != dim_mid")
            if self.w_freeze and self.dim_mid != self.dim_out:
                raise ValueError("Cannot freeze W(t) when dim_mid != dim_out")

            if not self.w_freeze:
                self.Wt = TimeTransformation(device, dim_mid, dim_out, False, w_freqs, interpolant=w_interpolant)
            else:
                self.Wt = None
            if not self.v_freeze:
                self.Vt = TimeTransformation(device, dim_in, dim_mid, False, v_freqs, interpolant=v_interpolant)
            else:
                self.Vt = None
            if not self.b_freeze:
                self.bt = TimeTransformation(device, dim_mid, dim_mid, True, b_freqs, interpolant=b_interpolant)
            else:
                self.bt = None

    def forward(self, t, x):
        out = x.transpose(0,1)
        if self.Vt is not None:
            out = self.Vt(t) @ out
        if self.bt is not None:
            out = out + self.bt(t).unsqueeze(-1)
        out = torch.relu(out) # torch.tanh(out)
        if self.Wt is not None:
            out = self.Wt(t) @ out
        
        return out.squeeze(0).transpose(0,1)

class ClassificationVFTimeShallow(nn.Module):
    def __init__(self, dim, device, width=128, nlayers=1, nfreqs=1, targets=None, interpolant='fourier', attractor_scale=1.0, attractor_strength=20.0):
        super().__init__()
        self.dim = dim
        self.width = width
        self.nlayers = nlayers
        self.nfreqs = nfreqs
        self.targets = targets
        self.attractor_strength = attractor_strength
        self.attractor_scale = attractor_scale
        #self.nlayers = nlayers

        if interpolant not in ['fourier', 'polynomial']:
            raise ValueError(f"Unknown interpolant: {interpolant}")
        self.interpolant = interpolant

        self.tsl_list = []
        
        # Add the input layer, if nlatyers == 1 then input and output are the same layer
        # so then no of V,W,b should be frozen
        self.tsl_list.append(TimeShallowLayer(device,
                            self.dim, self.width, self.width if self.nlayers > 1 else self.dim, 
                            w_freqs=nfreqs, v_freqs=nfreqs, b_freqs=self.nfreqs,
                            w_freeze=True if self.nlayers > 1 else False, v_freeze=False, b_freeze=False,
                            w_interpolant=self.interpolant, v_interpolant=self.interpolant, b_interpolant=self.interpolant
                        ))
        
        for _ in range(self.nlayers - 2):
            # For hidden layers, there is always a output layer or other hidden layer after it
            # so we should freeze W
            self.tsl_list.append(TimeShallowLayer(device,
                                self.width, self.width, self.width,
                                w_freqs=nfreqs, v_freqs=nfreqs, b_freqs=self.nfreqs,
                                w_freeze=True, v_freeze=False, b_freeze=False,
                                w_interpolant=self.interpolant, v_interpolant=self.interpolant, b_interpolant=self.interpolant
                            ))
        
        # Add the output layer
        if self.nlayers > 1:
            self.tsl_list.append(TimeShallowLayer(device,
                                self.width, self.width, self.dim,
                                w_freqs=nfreqs, v_freqs=nfreqs, b_freqs=self.nfreqs,
                                w_freeze=False, v_freeze=False, b_freeze=False,
                                w_interpolant=self.interpolant, v_interpolant=self.interpolant, b_interpolant=self.interpolant
                            ))
        
        self.tsl_list = nn.ModuleList(self.tsl_list)

        self.expscale = 3.0
        if self.targets is not None:
            # Find the shortest distance between any two targets
            tt = self.targets[0,...].transpose(0,1)
            distance_pairs = torch.cdist(tt, tt)
            distance_pairs.fill_diagonal_( float('inf') )
            min_dist = distance_pairs.min()
            # We wan't the exp to decay at maximum 1/4'th of the distance between targets
            min_dist = min_dist / 4.0
            self.expscale = self.attractor_scale / torch.square(min_dist)

    def forward(self, t, x):
        h = x
        for tsl in self.tsl_list:
            h = tsl(t, h)

        # Add target attractors
        if self.targets is not None:
            vdir = self.targets - x[...,None] # Pointing from x to targets
            d = torch.linalg.norm(vdir, dim=1)
            vdir = vdir / (torch.linalg.norm(vdir, dim=1, keepdim=True) + 1e-7)
            attractor_mag = self.attractor_strength * torch.exp(-self.expscale * torch.square(d))
            vdir = vdir * attractor_mag.unsqueeze(1)
            h = h + vdir.sum(dim=2)

        return h
