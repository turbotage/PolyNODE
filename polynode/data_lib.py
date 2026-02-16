import os
os.environ["SCIPY_ARRAY_API"]="1"
import torch
import scipy.optimize as optimize # used for uniform intrinsic sampling; elementwise is only usable if scipy.optimize is imported as a whole




# parameterized manifolds

class SphereND():
    """2d sphere in the space spanned by basis in n dimensional ambient space."""
    
    def __init__(self, x0, r, basis):
        """
        Build a 2d sphere in the space spanned by basis.
        
        There is no consitency check for basis or x0. 
        Scaling the basis effectively scales the radius. Non orthonormal basis yields elipsiods etc.

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            center.
        r : float
            radius.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            tuple of basis vectors with which the sphere is constructed. The default is (torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.]), torch.tensor([0.,0.,1.]) ).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.r = r 
        self.basis= basis
        return None
    def __call__(self,phi, theta):
        """
        Calculate a point on the sphere corresponding to the spherical coordinates 'phi' and 'theta'.

        Parameters
        ----------
        phi : torch.Tensor - shape (batch_size,)
            angle parameter in [0,2 pi).
        theta : torch.Tensor - shape (batch_size,)
            angle parameter in [0,pi).

        """
        x0_tensor=torch.stack([self.x0 for i in range(phi.shape[0])])
        return x0_tensor + self.r*(torch.outer(torch.cos(phi)*torch.cos(theta), self.basis[0]) + 
                                   torch.outer(torch.sin(phi)*torch.cos(theta), self.basis[1]) + 
                                   torch.outer(torch.sin(theta), self.basis[2])
                                   )

class PlaneND():
    """2d plane spanned by basis in n dimensional ambient space."""
    
    def __init__(self, x0, basis=(torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.])) ) :
        """
        Build a 2d plane spanned by basis in n dimensional ambient space.
        
        There is no consitency check for basis or x0. Scaling the basis effectively scales the parameters along the basis vectors. 
        Non orthonormal are basis possible yielding paralelograms when sampled on a grid.

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            offset.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            Tuple of basis vectors with which the hyperplane is constructed. The default is (torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.])).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.basis= basis
        return None
    def __call__(self,t, s):
        """
        Calculate a point on the plane corresponding to the parameters 's' and 't'.

        Parameters
        ----------
        t : torch.Tensor - shape (batch_size,)
            Parameter along the first basis vector.
        s : torch.Tensor - shape (batch_size,)
            Parameter along the second basis vector.

        """        
        x0_tensor=torch.stack([self.x0 for i in range(t.shape[0])])
        return x0_tensor +  torch.outer(t, self.basis[0])  + torch.outer(s, self.basis[1]) 
    

class SpiralND():
    """Archimedian 2d spiral in n dimensional ambient space."""
    
    def __init__(self, x0, speed_radius, speed_angle=1., phi_0=0., r_min=0.,
                 basis=(torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.]))):
        """
        Build an Archimedian 2d spiral in n dimensional ambient space.
        
        There is no consitency check for basis or x0. Scaling the basis effectively scales the radius. 
        Non orthonormal basis yields distored spirals.

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            center.
        r : float
            radius.
        speed_radius : float
            Speed of the radius growth.
        speed_angle : float, optional
            Angle speed. The default is 1..
        phi_0 : float, optional
            Angle offset. The default is 0..
        r_min : float, optional
            Minimal radius. The default is 0..
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            Tuple of basis vectors with which the spiral is constructed. The default is (torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.])).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.speed_radius = speed_radius
        self.speed_angle = speed_angle
        self.phi_0 = phi_0
        self.r_min = r_min
        self.basis= tuple(p.to(device=x0.device) for p in basis) # make sure default tensors are on the same device
        return None
    def __call__(self,phi):
        """
        Calculate aPint on the spiral corresponding to the angle parameter 'phi'.

        Parameters
        ----------
        phi : torch.Tensor - shape (batch_size,)
            Angle parameter.

        """
        r=self.r_min + self.speed_radius*phi
        x0_tensor = torch.stack([self.x0 for i in range(phi.shape[0])])
        phi_0_tensor = torch.tensor([self.phi_0 for i in range(phi.shape[0])])
        return (x0_tensor+ torch.outer( r*torch.cos(self.speed_angle*phi+ phi_0_tensor), self.basis[0]) + 
                 torch.outer( r*torch.sin(self.speed_angle*phi+ phi_0_tensor), self.basis[1]) )

class BoxND():
    """n dimensional box in n dimensional ambient space spanned by basis."""
    
    def __init__(self, x0, basis=( torch.tensor([1.,0.]), torch.tensor([0.,1.]))):
        """
        Build n dimensional box in n dimensional ambient space..

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            Corner of the box where the basis vectors start.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            List of basis vectors with which the box is constructed. The default is ( torch.tensor([1.,0.]), torch.tensor([0.,1.])).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.basis= tuple(p.to(device=x0.device) for p in basis)
        return None
    
    def __call__(self, *t):
        """
        Calculate a point in the box corresponding to the parameter list 't'.
        
        Use *args to workround the sampling implementation from dataset.

        Parameters
        ----------
        *t : list od length dim of torch.Tensor - shape (dim)
            List of parameters for the basis vectors.

        """
        return self.x0 + torch.sum( torch.stack( [torch.outer(t[i], self.basis[i]) for i in range(len(self.basis))] ) , dim = 0 ) 

class DiskND():
    """2 dimensional disc in n dimensional ambient space."""
    
    def __init__(self, x0, basis=(torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.]))):
        """
        Build 2 dimensional disc in n dimensional ambient space spanned by 'basis'.
        
        There is no consitency check for basis or x0. Scaling the basis effectively scales the radius. 
        Non orthonormal basis yields ellipse etc.

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            Center of the disc.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            Tuple of basis vectors that span the plane where the disc is constructed. The default is (torch.tensor([0.,1.,0.]), torch.tensor([1.,0.,0.])).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.basis= tuple(p.to(device=x0.device) for p in basis)
        return None
    def __call__(self,phi, r):
        """
        Calculate a point on the disc corresponding to the angular paramerters 'phi' and 'r'.

        Parameters
        ----------
        phi : torch.Tensor - shape (batch_size,)
            Angle parameter.
        r : torch.Tensor - shape (batch_size,)
            Radius parameter.
        """
        x0_tensor =torch.stack([self.x0 for i in range(phi.shape[0])])
        return x0_tensor+  torch.outer(r*torch.cos(phi), self.basis[0]) + torch.outer(r*torch.sin(phi+ 0), self.basis[1])

class PolynomialND():
    """1 dimensional polynomial graphical curve in n dimensional ambient space."""
    
    def __init__(self, x0, a, basis=(torch.tensor([0.,1.,0.]), torch.tensor([0.,0.,1.]))):
        """
        Build 1 dimensional polynomial graphical curve in n dimensional ambient space spanned by 'basis'.
        
        If p(t) is the polinomial at t, the graph is of the shape (p(t),t) on the standard basis (e1, e2).

        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            Offset in the ambiant space.
        a : list of floats
            List of coefficient for the polynomial cuve. Starting at degeree 0.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            Tuple of basis vectors that span the plane where the curve is constructed. 
            The first basis vecor is multiplied by the value of the polinomial, the second is multiplied by the parameter.
            The default is (torch.tensor([0.,1.,0.]), torch.tensor([0.,0.,1.])).

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.a = a
        self.basis= basis
        return None
    def __call__(self,t):
        """
        Calculate a point on the curve corsponding to the parameter 't'.

        Parameters
        ----------
        t : torch.Tensor - shape (batch_size)
            Curve parameter.
        """
        x0_tensor =torch.stack([self.x0 for i in range(t.shape[0])])
        return x0_tensor +  torch.outer(sum(self.a[i]*t**i for i in range(len(self.a))), self.basis[0]) + torch.outer(t, self.basis[1])



class DistanceSpiral():
    """Intrinsic distance function of the 2d spiral defined above. It can be calulated analytically."""
    
    def __init__(self, spiral):
        """
        Extract relevant parameters from 'spiral' to construct its intrisic distance function.

        Parameters
        ----------
        spiral : instnace of the Spiral_nd class
            Spiral.

        Returns
        -------
        None.

        """
        self.x0 = spiral.x0
        self.r_min = spiral.r_min
        self.speed_radius = spiral.speed_radius
        self.speed_angle = spiral.speed_angle
        
    def dist(self, phi):
        """
        Naive distance function.

        Parameters
        ----------
        phi : torch.Tensor - shape (batch_size,)
            Spiral angle parameter.

        """
        a= self.speed_angle*self.r_min/ self.speed_radius
        t= self.speed_angle * phi + a
        return self.speed_radius/ self.speed_angle /2 * (t**2 +1 )**(1/2) * t * torch.arctanh(t/(t**2 +1)**(1/2))
        
    def __call__(self, p):
        """
        Calculate intrinsinc distance for a point on the spiral from t=0, assuming orthonomal basis.
        
        There is no consitancy check if the point 'p' lies on the spiral.

        Parameters
        ----------
        p : torch.Tensor - shape (batch_size,)
            Point on the spiral
        """
        # get spiral angle parameter corresponding to p
        phi_p = (torch.linalg.vector_norm(p-self.x0, dim=1) - self.r_min)/self.speed_radius
        # claculate distnace from t0, note dist(0)!= 0 if r0>0
        return self.dist(phi_p) - self.dist( torch.zeros(phi_p.shape))


# transformations of R^n

class Shift():
    """Shift function on R^n, shift vector 'x_shift' specified at initialization."""
    
    def __init__(self, x_shift):
        self.x_shift = x_shift
        return None
    def __call__(self,x):
        """
        Shifts the input 'x' by the fixted vector 'x_shift'.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim)
            Point in R^n.
        """
        return x+torch.stack([self.x_shift for i in range(x.shape[0])])

class ReflectHyperplane():
    """Reflection on a hyperplane in R^n, hyperplane is specified at initialisation."""
    
    def __init__(self, x0, basis):
        """
        Initialize the hyperplane. The number of elements in 'basis' needs to be smaler or equal then the dimension of 'x0'.
        
        Parameters
        ----------
        x0 : torch.Tensor - shape (dim,)
            Base point of the hyperplane.
        basis : tuple of of length dim of torch.Tensor - shape (dim,), optional
            Basis that spans the hyperplane. Basis vectors are asumed to be linear independent.

        Raises
        ------
        RuntimeError
            Checks if the dimension of x0 is smaler or equal then the number of basis vectors.

        Returns
        -------
        None.

        """
        self.x0 = x0
        self.basis = basis
        dim = len(x0)
        if len(self.basis) >= dim: raise RuntimeError("Dimension mismatch!")
        return None
    def __call__(self,x):
        """
        Calculate the reflection of a point 'x' across the hyperplane.
        
        The normals of the hyperplane are calculated via singular value decomposition.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim)
            Point in R^n.

        """
        A= torch.stack(self.basis, dim =0)
        U, S, V = torch.linalg.svd(A)
        normals = V[len(self.basis):] # the singlular values are in decreasing order
        return x - 2*sum(   torch.outer((x-self.x0)@normal,  normal) for normal in normals)

class Scale():
    """Scale function in R^n, the scaleing factors 'factors' are set as a list during initialization."""
    
    def __init__(self, factors):
        self.factors = factors
        return None
    def __call__(self,x):
        """
        Calculate scaling of the input point 'x'. Each coordinate is scaled seperatly, according to the corresponding entry in 'factors'.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim)
            Point in R^n.
        """
        return  torch.stack([x[:,i]*self.factors[i] for i in range(x.shape[1]) ], dim=1)


class CompositionTrafo():
    """Composition for two instances of the transformation classes 'trafo_1' and 'trafo_2' as a new class."""
    
    def __init__(self, trafo_1, trafo_2):
        self.trafo_1 = trafo_1
        self.trafo_2 = trafo_2
        return None
    def __call__(self,x):
        """
        Apply the composition of trafo_1 and trafo_2 to 'x'.

        Parameters
        ----------
        x : torch.Tensor - shape (batch_size, dim)
            Point in R^n.
        """
        return self.trafo_1(self.trafo_2(x))
        

class DatasetSynth():
    """
    Class for genrating synthetic geometric data.
    
    The input data is sampled from the 'parameterization'. 
    The labels are generated from the input data via the 'transformation'.
    Assumes the 'domain' is eucledian.
    
    Attributes
    ----------
    parameterization : Instnace of a class for a geometric object from above.
    device : 
        pytorch device the data should be sent to.
    domain : list
        Domain where the parametrization is sampled on.
    dim_domain : int
        Dimension of the domain.
    transformation : Instance of transformation class from above.
        
    note : string
        A note to be saved in the automatic report describing the data. Default: ""
    sample_mode : sting
        Sampple = method
    sample_size = sample_size
    """
    
    def __init__(self, parameterization, domain, transformation, device=None, note=""):
        self.parameterization=parameterization
        self.device = device
        #self.device = domain.device
        self.domain = domain#.to(torch.device('cpu'))
        self.dim_domain = len(domain)
        self.transformation = transformation
        self.note=note
        return None
    
    def __make_dataset__(self, data_X):
        """
        Make a TensorDataset out of the sampled 'data_X' and the 'data_Y' obtained from 'data_X' by aplying the 'transformation'.

        Parameters
        ----------
        data_X : 
            Sampled geometric input data.

        Returns
        -------
        TensorDataset
            Synthetic data.

        """
        data_Y = self.transformation(data_X)       
        self.dataset = torch.utils.data.TensorDataset(
                            data_X if self.device is None else data_X.to(self.device), 
                            data_Y if self.device is None else data_Y.to(self.device)
                            )
        return self.dataset
    
    def sample(self, sample_size, method, **kwargs):
        """
        Sample the 'parametrisation' to generate synthetic data.
        
        Parameters
        ----------
        sample_size : int or list of int
            Number of sample points or number grid points specified for each dimension of the 'domain'.
        method : string
            Sample method to be used. Options are:
                "random": sample randomly, uniformly, indepently in every 'domain' dimension.
                "uniform": sample on a grid in the 'domain'.
                "1d_uniform_intrinsic": sample equidistantly according to the intrinsic distance of the 1 dimensional data manifold.
                                        Requires **kwargs: "dist" - the intrinsic distance function.    
        **kwargs : 
            "dist" : 
                The intrinsic distance function to be used for the "1d_uniform_intrinsic" method.

        Returns
        -------
        TensorDataset
            Synthetic data.

        """
        self.sample_mode = method
        self.sample_size = sample_size

        if self.sample_mode== "random":
            # Strip possible redundent structure.
            if type(self.sample_size) is not int: self.sample_size=self.sample_size[0]
            x_col=[]
            # Sample randomly and independly in the domain bounds.
            for i in range(self.dim_domain):
                lower_bound = torch.tensor([self.domain[i][0] for j in range(self.sample_size)])
                x_col.append( lower_bound + torch.rand(self.sample_size)* (self.domain[i][1]-self.domain[i][0]))
                
            data_X =self.parameterization(*x_col)
            return self.__make_dataset__(data_X)
        if self.sample_mode== "uniform":
            # Generate grid in the domain and transform the resulting tensor to apply 'parameterization' to the grid.
            x=[]
            for j  in range(self.dim_domain):
                x.append (  torch.linspace(self.domain[j][0], self.domain[j][1], self.sample_size[j] ))
                                            
            x_grid =torch.meshgrid(*x, indexing="ij") # x_grit is a tuple with the same length as x
            x_grid = tuple(x.flatten() for x in x_grid) # flatten every array in x_grid
            data_X = self.parameterization(*x_grid)
            return self.__make_dataset__(data_X)
        
        if self.sample_mode== "1d_uniform_intrinsic":
            device = self.domain[0].device
            dist= kwargs["dist"]
            # Calculate the bounds for the distance function form the bounds in the domain.
            bounds_dist =  dist(self.parameterization(self.domain[0])) 
            # Generate equidistant grid.
            target_dist_list = torch.linspace(bounds_dist[0], bounds_dist[1], sample_size[0])

            xl0 = self.domain[0][0]
            xr0 = self.domain[0][1]
            # Calculate the parameter list 't_sampled' coresponding to the equidistant grid points 'target_dist_list' numerically using a root finding algorithm.
            t_sampled = torch.zeros(target_dist_list.shape )
            for i in range(len(target_dist_list)):
                c =target_dist_list[i]
                def target(t):
                    # the parameterization needs a tensor on current device, but scipy/ numpy needs tensor on cpu to cast it to array
                    t=torch.tensor([t], device= device)
                    return  (c - dist(self.parameterization(t)) ).to(device="cpu")
                res = optimize.root_scalar(target, bracket= (xl0, xr0), method = "brentq")
                t_sampled[i] = res.root 
        # Evaluate the 'parametrization' on the calculated parameter list 't_sampled' to obtain the points on the data manifold.
        return self.__make_dataset__(self.parameterization(t_sampled))

    def save(self,path):
        """Save the dataset."""
        torch.save(self.dataset, path)
        return None
    
    def load(self,path):
        """Load a dataset."""
        print("Not implemented")
        return None
 
    
def get_time(hpara):
    """
    Generate a tensor containing the time steps for solving the flow, based on options set in the dict 'hpara'.
    
    hpara["ode_method"]: 'euler', 'midpoint', "heun2", "heun3",'rk4' need :
        hpara["t_end"] and hpara["time_steps"]
        the others (with automatic time stepping) need only hpara["t_end"]. 
        In this case 4 steps are created to be available for preproces and latent losses.
    """
    if hpara["ode_method"] in ['euler', 'midpoint', "heun2", "heun3",'rk4']: 
        t=torch.linspace(0., hpara["t_end"], hpara["time_steps"] )# prescribe time steps
    else: t = torch.tensor([0., hpara["t_end"]/4, hpara["t_end"]/2, hpara["t_end"]])  
    return t
    