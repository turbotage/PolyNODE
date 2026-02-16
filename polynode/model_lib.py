import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from torchdiffeq import odeint_adjoint, odeint

class CompressionFunction(torch.autograd.Function):
    """
    Compression function needed in the construction the compressing vector fields.
    
    It has its own class since we need to implement a custom backwards method to avoid divergence at 0.
    """
    
    @staticmethod
    def forward(ctx, input, alpha):
        """
        Forward pass, i.e. evalution, of the compression function with cutoff for small values of input.

        Parameters
        ----------
        input : torch.Tensor - shape (batch_size, dim_comp)
            directions of input coordinates to be compressed.
        alpha : float
            Hölder exponent in (0,1).

        Returns
        -------
        torch.Tensor - shape (batch_size, dim_comp)

        """        
        ctx.alpha = alpha
        ctx.a=   0 # 1e-7 #
        ctx.b=  1e-6 # 1e-9 #
        cutoff = CompVFBase.cutoff_transition( torch.abs(input), ctx.a, ctx.b) 
        compress = torch.sign(input) * torch.pow(torch.abs(input), alpha)
        ctx.save_for_backward(input, compress)

        return compress * cutoff

    @staticmethod
    def backward(ctx, grad_output):
        """
        Backward pass, i.e gradinent for chain rule, of the compression function with cutoff. Cutoff is introduced to avoid divergence at 0.

        Returns
        -------
        grad : torch.Tensor - shape (batch_size, dim_comp)

        """
        input, compress = ctx.saved_tensors
        alpha = ctx.alpha

        grad_input = grad_output.clone()
        product_rule = (
                        compress * CompVFBase.cutoff_transition_derivative(torch.abs(input), ctx.a, ctx.b) + 
                        CompVFBase.cutoff_hoelder_derivative(torch.abs(input), alpha, ctx.a, ctx.b)  
                        )
        grad = grad_input * product_rule

        return grad, None


class CompVFBase(nn.Module):
    """
    Base class for compresing vectorfield, used for the polyforld structure, bookeeping and some related methods.
    
    Cannot be used as an actual vector field in the model.
    Compressed directions refers to the coordinates that are maped to 0 by the flow before reaching the first stratification.
    Free directions refers to the coordinats where the vector field is not explicitely compressing.
    
    letting x be position
    x[:,0]                  is the direction of the stratification
    x[:,1:dim_comp+1]       are the compressing directions
    x[:,dim_comp+1:]        are the free directions

    
    Attributes
    ----------
    dim_comp : int, default 1
        dimension of the compressed coordinates
    dim_free : int, default 1
        dimension of the free coordinates
    dim : int, default 'dim_free' + 'dim_comp' +1 
        total dimension of the ambient space, +1 for the direction trough the stratification
        
    x1_free : float, default -3
        coordinate defining the hyper plane where the compressing vector field starts
    x1_free_buffer : float, default 'x1_free' -1
        first coordinate for the region where the transion between the free vector field and the compressing vector field takes place    
        
    x1_compressed : float, default 0
        first coordinate of the first stratification 
    x1_buffer_compress : float , default 'x1_compress' - 1 
        first coordinate for the region where the transion between compressing vector field and 0 takes place
        
    x1_decompressed : float, default 1
        first coordinate of the second stratification    
    x1_buffer_decompress : float, default 'x1_decompressed' +1
        first coordinate of the region where the transition from 0 to a free vector field takes place
    
    alpha : float, default 0.5
        Hölder exponent for the compression function.
    """
    
    def __init__(self, dim_free=1, dim_comp=1, **kwargs):
        """
        Initiallises the CompVf class.
        
        Parameters
        ----------
        dim_free : int, optional
            number of the free dimensions. The default is 1.
        dim_comp : int, optional
            number of compressed dimensions. The default is 1.
        **kwargs : 
            Parameters for the compresseion function and the polyfold structure. Set as attributes.
            
        Returns
        -------
        None.

        """
        super().__init__()
        self.idx_strat = 0
        self.idx_comp = 1
        self.idx_free = 1 + dim_comp

        self.dim_comp = dim_comp
        self.dim_free = dim_free
        self.dim = self.dim_free + self.dim_comp + 1 # +1 for thedirection trough the stratification
        
        try: 
            self.alpha = kwargs["alpha"]
        except KeyError: 
            self.alpha = 0.5
        if isinstance(self.alpha, torch.Tensor):
            print("alpha shouldn't be a tensor, retriving item")
            self.alpha = self.alpha.item()

        # geometry of the polyfold
        # cutoff the compressing vector field towards 0
        try: self.x1_compressed = kwargs["x1_compressed"]  # first stratification, end of transions
        except KeyError: self.x1_compressed = 0
        
        try: self.x1_buffer_compress = kwargs["x1_buffer_compress"] # start of transion , we want the data to be compressed by this point
        except KeyError: self.x1_buffer_compress = self.x1_compressed - 1
        
        # transition region for the free vector field to compressing vector field
        try: self.x1_free = kwargs["x1_free"] # end of transions
        except KeyError: self.x1_free = self.x1_compressed  -3

        try: self.x1_free_buffer = kwargs[""]  #  strat or transition
        except KeyError: self.x1_free_buffer = self.x1_free - 1
        
        # transition from 0 to free vector field
        try: self.x1_decompressed = kwargs["x1_decompressed"] # second point of stratification, start of transion
        except KeyError: self.x1_decompressed = 1
        
        try: self.x1_buffer_decompress = kwargs["x1_buffer_decompress"] # end of transion
        except KeyError: self.x1_buffer_decompress = self.x1_decompressed + 1
        

    @staticmethod
    def shift_scale(x, a, b):
        """
        Shifts the point 'x' by 'a' and scales by 'b-a'. used to build cutoff functions.
 
        Parameters
        ----------
        x : torch.Tensor
            input point.
        a : float
            shift parameter.
        b : float
            scale parameter.
            
        """
        return (x - a) / (b - a)
    
    @staticmethod
    def transition(x, flip=False):
        """
        Transition function with continous first derivative based on third degree polynomial.
        
        Parameters
        ----------
        x : torch.Tensor
            input coordinate.
        flip : bool, optional
            switch behavior at 0 and 1. The default is False.
            flip = False: from 0 at 0 to 1 at 1
            flip = True: from 1 at 0 to 0 at 1

        Returns
        -------
            value between 0 and 1.
            
        """
        x = torch.clamp(x, min=0, max=1)
        if flip:
            return  1 + (2*x - 3)*torch.square(x)
        else:
            return -(2*x - 3)*torch.square(x)
    
    @staticmethod
    def cutoff_transition(x, a=1e-7, b=1e-6, flip=False):
        """
        Transition function for 'x' between 'b' and 'a'.  Composition of the methods 'transition' and 'shift_scale'.

        Parameters
        ----------
        x : torch.Tensor
            input coordinate.
        a : float
            shift parameter. The default is 1e-7.
        b : float
            scale parameter. The default is 1e-6.
        flip : bool, optional
            switch behavior at 0 and 1. The default is False.
            flip = False: from 0 at 'a' to 1 at 'b'
            flip = True: from 1 at 'a' to 0 at 'b'

        Returns
        -------        
            value between 0 and 1.

        """       
        return CompVFBase.transition(CompVFBase.shift_scale(x, a, b), flip=flip)

    @staticmethod
    def cutoff_transition_derivative(x, a=1e-7, b=1e-6, flip=False):
        """Calculate derivative of the 'cut_oftransion' method. Has the same paramters as 'cut_oftransion'. Needed for the backward pass of the compression function."""
        x = CompVFBase.shift_scale(x, a, b)
        if flip:
            ret = 6*(x - torch.square(x)) / (b-a)
        else:
            ret = 6*(torch.square(x)-x) / (b-a)
        return torch.logical_and(x > a, x < b) * ret
    
    @staticmethod
    def cutoff_hoelder_derivative(x, alpha, a=1e-7, b=1e-6):
        """
        Calculate derivative of the compression function times the cutoff. Used in the "backward" method of the 'CompressionFunction' class.
        
        We may choose a=0 since the cutof function  decays faster then any hoelder exponent.
        But in that case we need to switch to to proper Hölder derivative manualy for x>b.
        If a>0 then we can clamp sutch that we never evaluate the singular hoelder derivative.
        Hence, we can handel the derivative in one term.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim_comp)
            input coordinate.
        alpha : float
            Hölder exponent.
        a : float
            shift parameter. The default is 1e-7.
        b : float
            scale parameter. The default is 1e-6.

        """
        if abs(a) <1e-15: 
            x_upper = torch.clamp(x, b, None)
            middle = (3*x/b - 2* torch.square(x/b)) * torch.pow(x, alpha)/b * (x<b)     
            hoelder_derivative = torch.pow( x_upper , alpha-1) * (x>=b)
            return alpha *(middle + hoelder_derivative)
        else:
            x_lower = CompVFBase.shift_scale(torch.clamp(x,a, b), a, b)
            x_upper = torch.clamp(x, a, None)
            return alpha * (3 * torch.square(x_lower) - 2 * torch.pow(x_lower,3)) * torch.pow( x_upper, alpha-1) 

    @staticmethod
    def compress(x, alpha):
        """
        Call the 'forward' pass of the compression function.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim_comp)
            directions of input coordinates to be compressed.
        alpha : float
            Hölder exponent.

        Returns
        -------
        torch.Tensor - shape (batch_size, dim_comp)

        """
        return CompressionFunction.apply(x, alpha)

    @abstractmethod
    def forward(self, t, x):
        pass


class CompVFReduced(CompVFBase):
    """
    Compressing vector field with neural networks and compressing structure.
    
    Based on the 'CompVFBase' class. With fixed x1 speed and without learned compression rates or latent dynamic.
    
    Attributes
    ----------    
    width : float
        width for all neural network layers. default: 50
    X1_speed : float
        fixed speed for the x1 compoment. default: 1

    compression_rate : list of floats
        fixed compression rate for each compressed coordinate. Passed in **kwargs. default: 25 for all coordinates
    
    X_free_left : nn.Sequential
        Simple sequential neural network.
    
    X_free_right : nn.Sequential
        Simple sequential neural network.
    """
    
    def __init__(self, dim_free=1, dim_comp=1, width=50, X1_speed = 1, **kwargs):
        """
        Initialize 'CompVFReduced' using the '__init__' of the base 'CompVFBase' with **kwargs passed. Initializes the neural networks.
        
        Parameters
        ----------
        dim_comp : int, default 1
            dimension of the compressed coordinates
        dim_free : int, default 1
            dimension of the free coordinates
        width : float
            width for all neural network layers. default: 50
        X1_speed : float, optional
            fixed speed for the x1 compoment. The default is 1.
        **kwargs :
            parameters for the compresseion function and the polyfold structure. Set as attributes..

        Returns
        -------
        None.

        """
        super().__init__(dim_free=dim_free, dim_comp=dim_comp, **kwargs)

        self.width = int(width)
        self.X1_speed = torch.tensor([X1_speed])

        self.compression_rate = torch.tensor(
            kwargs["compression_rate"] if "compression_rate" in kwargs 
            else [25 for i in range(self.dim_comp)]
        )

        
        self.X_free_left = nn.Sequential(
            nn.Linear(self.dim, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim -1, bias=False),
        )
        
        self.X_free_right = nn.Sequential(
            nn.Linear(self.dim, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim -1, bias=False),
        )
        

        for m in self.X_free_left.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)

        for m in self.X_free_right.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)
                   
    def forward(self, t, x):
        """
        Evaluate the vector field at time 't' and position 'x'.
        
        This function uses the neural networks of the class, the comression ,and cutoff functions 
        to build a compressing vector field that conforms to the polyfold structure 
        and that generates a flow which traverses the two stratifications.
        Several cutoff functions are used to either make sure the vector field is 0 away from the polyfold or
        glue together the vector field from different parts.

        Parameters
        ----------
        t : torch.Tensor (unused since this is not an time-dependent vector field)
            Time paramter.
        x : torch.Tensor - shape (batch_size, dim)
            Space pararameter.

        Returns
        -------
        vf_reduced : torch.Tensor - shape (batch_size, dim)
            Value of the vector field.

        """
        device=x.device
        # construct cutoff functions and assamble vector field without the x1 component
        shape_vf = (x.shape[0], self.dim-1)

        # turn off the free vector field on the left
        cutoff_free_left=torch.ones(shape_vf, dtype=torch.float32, device=device)
        # compressing direction        
        cutoff_free_left = CompVFBase.cutoff_transition(
                                                    x[:,0], 
                                                    self.x1_free_buffer, 
                                                    self.x1_free, 
                                                    flip=True
                                                ).unsqueeze(-1).expand(-1, self.dim-1)

        # turn on the free vector field on the right
        cutoff_free_right = CompVFBase.cutoff_transition(x[:,0], self.x1_decompressed, self.x1_buffer_decompress
                                                    ).unsqueeze(-1).expand(-1, self.dim-1)

        # components of the vector field in the compressing directions
        X_comp = torch.zeros(shape_vf, device=device)
        X_comp[:, :self.dim_comp] = torch.neg(self.compression_rate) * CompVFBase.compress(x[:,1:(1+self.dim_comp)], self.alpha)

        # cutoff for compressing part, we can cutoff for the free directions aswell since the X_comp is 0 there anyway
        # turn on compression
        cutoff_comp = CompVFBase.cutoff_transition(
                        x[:,0], 
                        self.x1_free_buffer, 
                        self.x1_free
                    )
        
        # turn of compression 
        cutoff_comp += CompVFBase.cutoff_transition(
                        x[:,0], 
                        self.x1_buffer_compress, 
                        self.x1_compressed,
                         flip=True
                    ) - 1 
        
        # The latent space is a part of hyperplane. This cutoff sets the vector field 0 away from that hyperplane
        compression_tolerance=1e-4 # genererous pointwise tolerence for cutoff, the models usually have an error in the latent space of about 1e-8
        cutoff_plane = CompVFBase.cutoff_transition(
                        torch.linalg.vector_norm(x[:, 1: (1+ self.dim_comp)], dim=1, ord=2), 
                        compression_tolerance*0.1, 
                        compression_tolerance,
                        flip=True
                    )
        # rectrict the cutoff from the hyperplane above to the latent region with a tollerance of 0.001 on either side
        cutoff_restriction = (CompVFBase.cutoff_transition(
                                x[:,0],
                                self.x1_compressed,
                                self.x1_compressed+0.001,
                                flip=True
                            ) + CompVFBase.cutoff_transition(
                                x[:,0],
                                self.x1_decompressed-0.001,
                                self.x1_decompressed,
                                flip=False
                            ))
        cutoff_restriction += (1 - cutoff_restriction) * cutoff_plane
        
        # build the final vector field from the pieces above
        vf_reduced = torch.zeros(x.shape, device=device)
        vf_reduced[:,0] = self.X1_speed
        vf_reduced[:,1:] = (self.X_free_left(x)*cutoff_free_left +  
                            self.X_free_right(x)*cutoff_free_right + 
                            X_comp*cutoff_comp.unsqueeze(-1)) * cutoff_restriction.unsqueeze(-1)
        
        return vf_reduced
    
class FlowModel(nn.Module):
    """
    Combines a vector filed with an ODE solver for ease of use.
    
    Attributes
    ----------    
    width : float
        width for all neural network layers. default: 50
    comp_vf : 
        instance of a vector filed that can be handeled by integration method 
    method : 
        odeint function
    rtol : float
        relative toletrance for the ode intergration method
    atol : float
        relative toletrance for the ode intergration method

    """
    
    def __init__(self, comp_vf, hpara):
        """
        Initialize the Flow_Mol class and set attributes.

        Parameters
        ----------
        comp_vf : 
            Vector field.
        hpara : dict
            dict containing information about the ODE solver:
                hpara["ode_method"] - an odeint function
                hpara["rtol"]
                hpara["atol"]

        Returns
        -------
        None.

        """
        super().__init__()
        self.comp_vf = comp_vf
        self.method = hpara["ode_method"]
        self.rtol = hpara["rtol"]
        self.atol = hpara["atol"]

        if hpara["use_ode_adjoint"]:
            self.odeint = odeint_adjoint
        else:
            self.odeint = odeint
        if hpara["use_projected_odeint"]:
            self.odeint = proj_odeint(odeint)

    def forward(self, t, x):
        """
        Calculate the value of the flow at time 't' stating at 'x' by integrating the vector field with the specified 'method'.

        Parameters
        ----------
        t : torch.Tensor - shape (num_timesteps,)
            Flow time parameter.
        x : torch.Tensor - shape (batch_size, dim)
            Start point of the flow.

        Returns
        -------
        float
            Value of the flow at time 't' stating at 'x'.

        """
        return self.odeint(self.comp_vf, x, t, 
                method=self.method, rtol=self.rtol, atol=self.atol)
    

def proj_odeint(odeint_old):
    """
    Piece-wise odeint with projection in latent region.
    
    Assumes the vector field is setup such that flow is in the latent space after half the time.
    
    Parameters
    ----------
    odeint_old : 
        odeint function to be used on both halves of the integration.

    Returns
    -------
        odeint function.
        
    """    
    def odeint(X, y0, t, method, rtol, atol):
        t_half_1= t[:t.shape[0]//2 +1 ]
        t_half_2= t[t.shape[0]//2:]
        y1 = odeint_old(X, y0, t_half_1, method=method,
                              rtol=rtol, atol=atol)
        y_filter = (y1[:,:,0] < X.x1_compressed+ 0.1 ) + (y1[:,:,0] > X.x1_decompressed -0.1)
        projection = torch.ones(y1.shape )
        projection[:,:,1:1+X.dim_comp] = y_filter.unsqueeze(-1)
        y1 = y1*projection
        y2 = odeint_old(X, y1[-1], t_half_2, method=method, rtol=rtol, atol=atol)
        return torch.cat([y1,y2[1:]], dim=0)
    return odeint

class CompVF(CompVFBase):
    """
    Compressing vector field with neural networks and compressing structure. Based on the 'CompVFBase' class.
    
    Attributes
    ----------    
    width : float
        width for all neural network layers, default: 50
        
    X1_speed : float
        fixed speed for the x1 compoment. default: 1

    k_lower_bound : float
        lower bound for all compressing rates, default: 

    k_upper_bound : float
        upper bound for all compressing rates, default: 
            
    k_comp : nn.Sequential
        neural network for the compression rates
        
    X_free_left : nn.Sequential
        neural network for the region left of the first stratification
    
    X_free_right : nn.Sequential
        neural network for the region right of the second stratification
    X_free_latent : nn.Sequential
        neural network for the latent region between the two stratifications   
    """
    
    def __init__(self, dim_free=1, dim_comp=1,  width=50, X1_speed = 1, **kwargs):
        """
        Initialize 'CompVF' using 'CompVFBase' with **kwargs passed. Initializes the neural networks.

        Parameters
        ----------
        dim_free : int, optional
            DESCRIPTION. The default is 1.
        dim_comp : int, optional
            DESCRIPTION. The default is 1.
        width : int, optional
            DESCRIPTION. The default is 50.
        X1_speed : float, optional
            fixed speed for the x1 compoment. The default is 1.
        **kwargs :

        Returns
        -------
        None.

        """
        super().__init__(dim_free=dim_free, dim_comp=dim_comp, **kwargs)

        self.width = int(width)
        self.X1_speed = torch.tensor([X1_speed])

        self.k_lower_bound = torch.tensor(
            kwargs["k_lower_bound"] if "k_lower_bound" in kwargs
            else [10 for i in range(self.dim_comp)]
        )

        self.k_upper_bound = torch.tensor(
            kwargs["k_upper_bound"] if "k_upper_bound" in kwargs
            else [25 for i in range(self.dim_comp)]
        )
        
        
        self.X_free_left = nn.Sequential(
            nn.Linear(self.dim, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim-1, bias=False),
        )
        
        self.X_free_right = nn.Sequential(
            nn.Linear(self.dim, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim-1, bias=False),
        )
        
        self.X_free_latent = nn.Sequential(
            nn.Linear(self.dim_free + 1, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim_free, bias=False),
        )
        
        # models the rates for the compressing directions
        self.k_comp = nn.Sequential(
            nn.Linear(self.dim, self.width),
            nn.Tanh(),
            nn.Linear(self.width, self.dim_comp, bias=False),
            nn.Sigmoid()
        )
        
        
        for m in self.k_comp.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)

        for m in self.X_free_left.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)

        for m in self.X_free_right.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)
                
        for m in self.X_free_latent.modules():
            if isinstance(m, nn.Linear):
                std=1/m.in_features**(1/2)
                nn.init.normal_(m.weight, mean=0, std=std)
                if m.bias is not None: nn.init.constant_(m.bias, val=0)

    def forward(self, t, x):
        """
        Evaluate the vector field at time 't' and position 'x'.
        
        This function uses the neural networks of the class, the comression ,and cutoff functions 
        to build a compressing vector field that conforms to the polyfold structure 
        and that generates a flow which traverses the two stratifications.
        Several cutoff functions are used to either make sure the vector field is 0 away from the polyfold or
        glue together the vector field from different parts.

        Parameters
        ----------
        t : torch.Tensor (unused since this is not an time-dependent vector field)
            Time paramter.
        x : torch.Tensor - shape (batch_size, dim)
            Space pararameter.

        Returns
        -------
        vf_reduced : torch.Tensor - shape (batch_size, dim)
            Value of the vector field.

        """
        device=x.device
        # construct cutoff functions and assamble vector field without the x1 component
        shape_vf = (x.shape[0], self.dim-1)
        
        # turn off the free vector field on the left        
        cutoff_free_left=torch.ones(shape_vf, dtype=torch.float32, device=device)
        # compressing direction        
        cutoff_free_left[:,:self.dim_comp] = CompVFBase.cutoff_transition(
                                                    x[:,0], 
                                                    self.x1_free_buffer, 
                                                    self.x1_free, 
                                                    flip=True
                                                ).unsqueeze(-1).expand(-1, self.dim_comp)
        # free direction
        cutoff_free_left[:,self.dim_comp:] = CompVFBase.cutoff_transition(
                                                    x[:,0], 
                                                    self.x1_buffer_compress, 
                                                    self.x1_compressed, 
                                                    flip=True
                                                ).unsqueeze(-1).expand(-1, self.dim_free)

        # turn on the free vector field in the latent space
        cutoff_free_latent=torch.zeros(shape_vf, dtype=torch.float32, device=device)
        cutoff_free_latent = CompVFBase.cutoff_transition(
                                                    x[:,0], 
                                                    self.x1_buffer_compress, 
                                                    self.x1_compressed
                                                    ).unsqueeze(-1).expand(-1, self.dim-1)

        # turn off the free vector field in latent space
        cutoff_free_latent = cutoff_free_latent + CompVFBase.cutoff_transition(x[:,0], self.x1_decompressed, 
                                                            self.x1_buffer_decompress, flip=True
                                                            ).unsqueeze(-1).expand(-1, self.dim-1) -1

        # turn on the free vector field on the right
        cutoff_free_right = CompVFBase.cutoff_transition(x[:,0], self.x1_decompressed, self.x1_buffer_decompress
                                                    ).unsqueeze(-1).expand(-1, self.dim-1)

        # components of the vector field in the compressing directions
        X_comp = torch.zeros(shape_vf, device=device)
        rate = self.k_comp(x)*(self.k_upper_bound-self.k_lower_bound) + self.k_lower_bound 
        X_comp[:,:self.dim_comp] = torch.neg(rate) * CompVFBase.compress(x[:,1:(1+self.dim_comp)], self.alpha)

        # cutoff for compressing part, we can cutoff for the free directions aswell since the X_comp is 0 there anyway
        # turn on compression
        cutoff_comp = CompVFBase.cutoff_transition(
                        x[:,0], 
                        self.x1_free_buffer, 
                        self.x1_free
                    )
        
        # turn of compression 
        cutoff_comp += CompVFBase.cutoff_transition(
                        x[:,0], 
                        self.x1_buffer_compress, 
                        self.x1_compressed,
                         flip=True
                    ) - 1 
        
        # The latent space is a part of hyperplane. This cutoff sets the vector field 0 away from that hyperplane
        compression_tolerance=1e-5 # genererous pointwise tolerence for cutoff, the models usually have an error in the latent space of about 1e-8
        cutoff_plane = CompVFBase.cutoff_transition(
                        torch.linalg.vector_norm(x[:, 1: (1+ self.dim_comp)], dim=1, ord=2), 
                        compression_tolerance*0.1, 
                        compression_tolerance,
                        flip=True
                    )
        
        cutoff_restriction = (CompVFBase.cutoff_transition(
                                x[:,0],
                                self.x1_compressed,
                                self.x1_compressed+0.001,
                                flip=True
                            ) + CompVFBase.cutoff_transition(
                                x[:,0],
                                self.x1_decompressed-0.001,
                                self.x1_decompressed,
                                flip=False
                            ))
        # rectrict the cutoff from the hyperplane above to the latent region with a tolerance of 0.001 on either side
        cutoff_restriction += (1 - cutoff_restriction) * cutoff_plane
        
        # add 0s for the compressed dimensions to the vector field in the latent space
        X_free_latent = torch.zeros(shape_vf, device=device)
        # prepare the point x to work as in input to 'X_free_latent' by deleteing the components in the compressing directions
        x_latent=torch.cat([x[:, 0].unsqueeze(-1), x[:, self.dim_comp + 1:]], dim=-1) 
        X_free_latent_reduced = self.X_free_latent(x_latent)
        # build the vectorfield in the latent space
        X_free_latent[:, 0] = X_free_latent_reduced[:, 0]
        X_free_latent[:, self.dim_comp + 1:] = X_free_latent_reduced[:, 1:]
    
        # build the final vector field from the pieces above, with constant speed in the first component
        vf = torch.zeros(x.shape, device=device)
        vf[:,0] = self.X1_speed
        vf[:,1:] = (self.X_free_left(x)*cutoff_free_left 
                    + X_free_latent*cutoff_free_latent 
                    + self.X_free_right(x)*cutoff_free_right 
                    + X_comp*cutoff_comp.unsqueeze(-1)) * cutoff_restriction.unsqueeze(-1)
        return vf
    