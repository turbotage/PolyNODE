import torch
import torch.nn as nn

def intrinsic_mse(p, q, x_min, x_max, dim_comp, reduction = "mean"):
    """
    Calculate the intrinsic mse error between 'p' and 'q' based on the energy-optimal piece wise path through the polyfold.
    
    This construction reduces to the normal mse if 'p' and 'q' are either both left of 'x_min' or right of 'x_max'.
    Assumes dim 0 is batch dimension and points are structured (strat, compression, free).
    Note: the square of the intrinsic distance is not differentiable.

    Parameters
    ----------
    p : torch.Tensor - shape (batch_size, dim)
        input point.
    q : torch.Tensor - shape (batch_size, dim)
        input point.
    x_min : float
        start of the latent space of the polyfold.
    x_max : float
        end of the latent space of the polyfold.
    dim_comp : int
        Number of dimensions of the compressed coordinates.
    reduction : string, optional
        Option for using the mean across a batch. The default is "mean".

    """
    #tensorize x_min and x_max 
    x_min = torch.tensor([x_min for i in range(p.shape[0])])
    x_max = torch.tensor([x_max for i in range(p.shape[0])])
    
    mse=nn.MSELoss(reduction=reduction)
    
    # sort p and q for the intermediate point construction
    p_temp = p * (p[:,0] <= q[:,0]).unsqueeze(-1) + q * (q[:,0] < p[:,0]).unsqueeze(-1)
    q = q * (q[:,0] >=  p[:,0]).unsqueeze(-1)  + p * (p[:,0] > q[:,0]).unsqueeze(-1)
    p = p_temp
    
    # build intermediate points in latent space, a_mid and b_mid without the free dimensions. 
    # The free dimensions are taken care of seperatly
    shape_reduced=(p.shape[0], dim_comp+1)
    a_mid = torch.zeros(shape_reduced)
    a_mid[:,0]= x_min
    b_mid = torch.zeros(shape_reduced)
    b_mid[:,0]= x_max
    
    p_reduced = p[:, : 1+dim_comp]
    q_reduced = q[:, : 1+dim_comp]
    p_free= p[:, 1+dim_comp :]
    q_free= q[:, 1+dim_comp :]
    # turn on intermediate points based on relative position of p,q, x_min, x_max
    a=( p_reduced*(p[:,0] >= x_min).unsqueeze(-1) + q_reduced*(q[:,0] <= x_min).unsqueeze(-1) 
       + a_mid*( (p[:,0] < x_min)*(q[:,0] > x_min) ).unsqueeze(-1) )
    b=( q_reduced*(q[:,0] <= x_max).unsqueeze(-1) + p_reduced*(p[:,0] >= x_max).unsqueeze(-1) 
       + b_mid*( (q[:,0] > x_max)*(p[:,0] < x_max) ).unsqueeze(-1) )
    
    # need to check if free dimensions are present, else mse yields nan
    if q_free.shape[1] ==0:
        return mse(p_reduced,a) + mse(a,b) + mse(b,q_reduced)
    else:
        # technically the first 3 terms need a factor in some cases; namely the number of breaks in the curve;
        # but this is inconsistent in the masking construction and numerically irrelevant for our purposes
        return mse(p_reduced,a) + mse(a,b) + mse(b,q_reduced) + mse(p_free, q_free)
    
    

class LossLatent():
    """
    Custom loss function class based on MSELoss.
    
    Assumes 'y_pred' passed in '__call__' has  at least three entires in the time dimention.
    
    Attributes
    ----------
    x_min : float
        start of the latent space of the polyfold.
    x_max : float
        end of the latent space of the polyfold.
    weight :  float
        weight for the loss at half the time.
    reduction : string
        Option for using the mean across a batch. The default is "mean".
    """
    
    def __init__(self, x_min, x_max, weight=2, reduction="mean"):
        self.x_min = x_min
        self.x_max = x_max
        self.weight = weight
        self.reduction= reduction
        return None
    def __call__(self,y_pred,y_target,t):
        """
        Calculate the loss between 'y_pred' and 'y_target' using the pytorch MSELoss plus an extra term if 'y_pred' at half the time is not in the latent space.

        Parameters
        ----------
        y_pred : torch.Tensor - shape (time steps, batch_size, dim)
            Tensor predicted by the flow. Assumes first dimension is integration time.
        y_target : torch.Tensor - shape (batch_size, dim)
            Target tensor.
        t : torch.Tensor - shape (times steps,)
            Time.


        """
        t_mid=int(len(t)/2)
        x_min= torch.tensor([self.x_min for i in range(len(y_pred[t_mid]))])
        x_max= torch.tensor([self.x_max for i in range(len(y_pred[t_mid]))])
        # the additional term val is only non zero if 'y_pred' is not in the latent space. 
        # In this case we calulate the distance of the first component to either stratification point.
        val= ( torch.abs(y_pred[t_mid,:,0]-x_min) * (y_pred[t_mid,:,0] <self.x_min) 
              + torch.abs(y_pred[t_mid,:,0]-x_max) * (y_pred[t_mid,:,0]>self.x_max) )
        return ( self.weight * nn.MSELoss(reduction=self.reduction)(val, torch.zeros(y_pred[t_mid,:,0].shape)) 
                + nn.MSELoss(reduction=self.reduction)(y_pred[-1], y_target))


class LossPreprocessIsometry():
    """
    Custom loss function class based on 'intrinsic_mse' with preprocessing target instead of latent target and approximate isometry forcing.
    
    Assumes one dimensional, topologicaly trivial data.
    Assumes the input batch is randomized for the approximate isometry loss term.
    
    Attributes
    ----------
    x_min : float
        start of the latent space of the polyfold.
    x_max : float
        end of the latent space of the polyfold.
    dim_comp : int
        Number of dimensions of the compressed coordinates.
    dist : 
        Intrinsic distance function on the input data manifold.
    x1_pre : float
        Preprocessing target x1 coordinate.
    x_pre_target : float
        Preprocessing target value for compressing coordinates. Default: 1
    weight :  float
        weight for the latent and isometry term
    iso_scale : float
        Scale for the distance in the free direction. Default: 1
    reduction : string
        Option for using the mean across a batch. The default is "mean".    
    
    """
    
    def __init__(self, x_min, x_max, dim_comp, dist, x1_pre = -2., x_pre_target = 1., weight=1, iso_scale = 1, reduction="mean"):
        self.x_min = x_min
        self.x_max = x_max
        self.dim_comp = dim_comp
        self.dist = dist
        self.x1_pre = x1_pre
        self.x_pre_target = x_pre_target
        self.weight = weight
        self.iso_scale = iso_scale
        self.reduction= reduction
        return None
    
    def __call__(self,y_pred,y_target, t):
        """
        Calculate the loss.

        Parameters
        ----------
        y_pred : torch.Tensor - shape (times steps, batch_size, dim)
            Tensor predicted by the flow. Assumes first dimension is integration time.
        y_target : torch.Tensor - shape (batch_size, dim)
            Target tensor for the final time
        t : torch.Tensor - shape (time steps,)
            Time.


        """
        # get start points
        y0 = y_pred[0]
        # time for the end of preprocessing and prediction at that time
        t_pre = int(len(t)/4)  
        y_pre = y_pred[t_pre,:, : 1+self.dim_comp]
        # build the preprocessing target tensor
        y_target_pre = torch.zeros( (y_target.shape[0], 1+self.dim_comp) )
        y_target_pre[:,:] = self.x_pre_target
        y_target_pre[:,0] = self.x1_pre     
        # calculate defect of the distances on the shifted input and result
        # since we assume the batch is random the shift in the batch dimensions compares random points
        y_pre_dist = torch.linalg.vector_norm(y_pred[t_pre, 0:-1, 1+self.dim_comp: ] - y_pred[t_pre, 1:, 1+self.dim_comp : ], dim=-1)
        if self.reduction == "mean":
            isometry_term =torch.sum( torch.abs(torch.abs(self.dist(y0[:-1]) - self.dist(y0[1:])) - self.iso_scale*y_pre_dist)  ) / len(y_pred[0])
        else:
            isometry_term =torch.sum( torch.abs(abs(self.dist(y0[:-1]) - self.dist(y0[1:])) - self.iso_scale*y_pre_dist) )
            
        return ( self.weight* intrinsic_mse(y_pre, y_target_pre, self.x_min, self.x_max, self.dim_comp, reduction=self.reduction) + 
                 self.weight* isometry_term  +
                 intrinsic_mse(y_pred[-1], y_target, self.x_min, self.x_max, self.dim_comp, reduction=self.reduction) )
