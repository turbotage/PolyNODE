import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
plt.rcParams['text.usetex'] = True
import numpy as np
import csv
# for extraction of tensors form saved strings
import re 
import ast 

plasma = mpl.colormaps["plasma"] # manualy get the color map to get around automatic rescaling

# Default setup for solver and optimizer
hpara={} 
hpara["gpu"] = 0
hpara["device"] =  torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')  #torch.device('cuda:' + str(hpara["gpu"]) if torch.cuda.is_available() else 'cpu')
hpara["ode_method"] = "euler"       
hpara["use_ode_adjoint"] = True  
hpara["use_projected_odeint"] = True
hpara["rtol"] = 1e-4 
hpara["atol"] = 1e-5
hpara["start_lr"] = 1e-3 
hpara["momentum"] = 0.3
hpara["lr_end_factor"] = 0.1
hpara["lr_end_epochs"] = 2000
hpara["patience"]= 25
hpara["max_epochs"] = 20000
hpara["batch_size"] = 10000
hpara["num_workers"] = 0
hpara["test_freq"] =50
hpara["t_end"] = 1.
hpara["test"] = False


class EarlyStopper:
    """Automatic stopper for the training of the neural networks."""
    
    def __init__(self, patience=1, relative_acc=0.001, absolute_acc = 1e-10):
        """
        Initilize the EarlyStopper.

        Parameters
        ----------
        patience : int, optional
            Number of calls without significant improovment until the training is stoped. The default is 1.
        relative_acc : float, optional
            Relative accuracy compared to previous best loss to register improovment. The default is 0.001.
        absolute_acc : float, optional
            Absolute accuracy threshold blow which the training is always stopped. The default is 1e-10.

        Returns
        -------
        None.

        """
        self.patience = patience
        self.absolute_acc = absolute_acc
        self.relative_acc = relative_acc
        self.counter = 0
        self.min_validation_loss = float('inf')

    def __call__(self, validation_loss):
        """
        Compare 'validation_loss' to previous best loss, using 'relative_acc', and 'absolute_acc' to decide if the training should be stopped.

        Parameters
        ----------
        validation_loss : float
            Input loss.

        Returns
        -------
        bool
            True - stopp the training.
            False - continue.

        """
        if validation_loss < self.min_validation_loss*(1-self.relative_acc) :
            self.min_validation_loss = validation_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                print("Training stagnated! Patience: ", self.patience, "\n")
                return True
        if self.absolute_acc != None: 
            if validation_loss < self.absolute_acc:
                print("Training Stopped! Validation loss less then precribed absolute value: ", self.absolute_acc, "\n")
                return True
        return False
 
  
# plots and reporting, matplotlib uses numpy arrays internaly, and only tensors on the cpu can be converred to numpy arrays
def plot_data(data_list, path =None, format = "png", **kwargs):
    """
    Plot data in 'data_list' insterpreted as points in R^2 or R^3. Using scatter from matplotlib.
    
    If the points are in R^n for n> 3 then only the first two and the last coordinates are plotted.
    The colormap function uses data order as colormap, making the color meaningless for randomized data.

    Parameters
    ----------
    data_list : list of torch.Tensor - shape (batch size, dim)
        List of time slices.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        image format. The default is "png".
    **kwargs : 
        Options passed to matplotlib:
            "title" - default: ""
            "labels" - default: [None for i in range(len(data_list))]
            "color" - default: [None for i in range(len(data_list))]
            "use_color_map" - default: False
        

    Returns
    -------
    None.

    """    
    data_list = [d.to(device="cpu") for d in data_list]
    try: title = kwargs["title"]
    except KeyError: title =""
    
    try: labels = kwargs["labels"]
    except KeyError: labels = [None for i in range(len(data_list))]
    
    try: color = kwargs["color"]
    except KeyError: color = [None for i in range(len(data_list))]
    
    try: use_color_map = kwargs["use_color_map" ]
    except KeyError: use_color_map= False
    
    
    fig = plt.figure()    
    if len(data_list[0][0])==2:
        ax = fig.add_subplot()

        for j in range(len(data_list)):
            xs = data_list[j][:,0]
            ys = data_list[j][:,1] 
            ax.scatter(xs, ys, marker=".", label=labels[j], c=color[j])
        if labels[0] != None: ax.legend()
        ax.set_xlabel(r'$x_1$')
        ax.set_ylabel(r'$y_1$')
        
    else:
        ax = fig.add_subplot(projection='3d')
        
        for j in range(len(data_list)):
            xs = data_list[j][:,0]
            ys = data_list[j][:,-1]
            zs = data_list[j][:,1]
            if use_color_map:
                c_function = np.linspace(0,1,len(data_list[j]))
                ax.scatter(xs, ys, zs, marker=".", label=labels[j], c=c_function,cmap = "plasma")
            else: ax.scatter(xs, ys, zs, marker=".", label=labels[j], c=color[j])
        if labels[0] != None: ax.legend()
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$x_1$')
        ax.set_zlabel(r'$y_1$')
        
    ax.set_title(title)
    if path != None: plt.savefig(path,  format=format, pad_inches=0.35, bbox_inches='tight')
    plt.show()     
    plt.close()
    return None

def c_function_xfree(data, x_min, x_max, colormap=plasma):
    """
    Colormap for the elements of 'data' scaling the color according to 'x_min' and 'x_max'. Used to color the fourth dimension in the 4d plots.

    Parameters
    ----------
    data : torch.Tensor - shape (time steps, batch size, dim)
        .
    x_min : float
        start of the colormap.
    xfree_xmax : float
        end of the colormap.
    colormap : TYPE, optional
        Colormap to color 'data'. The default is plasma.

    """
    if x_min!=x_max:
        return colormap( (data[:,-2] - x_min)/(x_max - x_min) )
    else:
        return colormap( torch.zeros(data[:,-2].shape) )

def plot_data_4d(data_list, path =None, format = "png", **kwargs):
    """
    Plot data in 'data_list' insterpreted as points in R^4. Using scatter from matplotlib.
    
    Only the first two and the last coordinates are plotted. The third coordinate is represented via a color map.

    Parameters
    ----------
    data_list : list of torch.Tensor - shape (batch size, dim)
        List of time slices.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        image format. The default is "png".
    **kwargs : 
        Options passed to matplotlib:
            "title" - default: ""
            "labels" - default: [None for i in range(len(data_list))]
            "color" - used only if no colormap is specified. default: [None for i in range(len(data_list))]
            "use_color_map" - default: False
            "xlim" - limits for the x coordinate
            "ylim" - limits for the y coordinate
            "zlim" - limits for the z coordinate

    Returns
    -------
    None.

    """    
    data_list = [d.to(device="cpu") for d in data_list]
    try: title = kwargs["title"]
    except KeyError: title =""
    
    try: labels = kwargs["labels"]
    except KeyError: labels = [str(i) for i in range(len(data_list))]
    
    try: color = kwargs["color"]
    except KeyError: color = [None for i in range(len(data_list))]
    
    try: use_color_map = kwargs["use_color_map" ]
    except KeyError: use_color_map= False
    
    xfree_min = min(  min( data[:,-2]) for data in data_list)
    xfree_xmax = max(  max( data[:,-2]) for data in data_list)
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    
    for j in range(len(data_list)):
        xs = data_list[j][:,0]
        ys = data_list[j][:,-1]
        zs = data_list[j][:,1]
        
        if use_color_map:
            cm = c_function_xfree(data_list[j], xfree_min, xfree_xmax, colormap=plasma)
            ax.scatter(xs, ys, zs, marker=".", label=labels[j], c=cm)
        else: ax.scatter(xs, ys, zs, marker=".", label=labels[j], c=color[j])
        
    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'$y_1$')
        
    try: 
        ax.set_xlim(*kwargs["xlim"]) 
    except KeyError:
        pass
    try: 
        ax.set_ylim(*kwargs["ylim"]) 
    except KeyError:
        pass
    try: 
        ax.set_zlim(*kwargs["zlim"]) 
    except KeyError:
        pass
    
    ax.set_title(title)
    if path != None: plt.savefig(path, format=format, pad_inches=0.35, bbox_inches='tight')
    plt.show()     
    plt.close()
    return None
    

def plot_trajectrories_4d(y, path =None, format = "png", n_trajectories=13, **kwargs):
    """
    Plot selected trajectories of 'y' insterpreted as points in R^4, using scatter from matplotlib.
    
    'y' is assumed to be the putput of the flow applied to a batch of points. The first dimension of 'y'is assmed to correspond to time, 
    the second is the batch dimension, the third corresponds to the spacial dimensions.
    Only the first two and the last spacial coordinates are plotted. The third coordinate is represented via a color map.

    Parameters
    ----------
    y : torch.Tensor - shape (time steps, batch size, dim)
        flow.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        image format. The default is "png".
    n_trajectories : int, optional
        Number of trajectories. The default is 13.
    **kwargs : 
        Options passed to matplotlib:
            "title" - default: ""
            "xlim" - limits for the x coordinate
            "ylim" - limits for the y coordinate
            "zlim" - limits for the z coordinate

    Returns
    -------
    None.

    """
    try: title = kwargs["title"]
    except KeyError: title =""
    
    step = y.shape[1]//n_trajectories
    y= y[:,::step,:]
    xfree_min = min(y[:,:,-2].flatten())
    xfree_xmax = max(y[:,:,-2].flatten())

    fig = plt.figure()

    ax = fig.add_subplot(projection='3d')
    for i in range(y.shape[1]):
        xs = y[:,i,0]
        ys = y[:,i,-1]
        zs = y[:,i,1]
        cm = c_function_xfree(y[:,i], xfree_min, xfree_xmax)
        ax.scatter(xs, ys, zs, marker=".",c=cm)

    ax.set_xlabel(r'$\tau$')
    ax.set_ylabel(r'$x_1$')
    ax.set_zlabel(r'$y_1$')
        
    try: 
        ax.set_xlim(*kwargs["xlim"]) 
    except KeyError:
        pass
    try: 
        ax.set_ylim(*kwargs["ylim"]) 
    except KeyError:
        pass
    try: 
        ax.set_zlim(*kwargs["zlim"]) 
    except KeyError:
        pass


    ax.set_title(title)
    if path != None: plt.savefig(path, format=format, pad_inches=0.35, bbox_inches='tight')
    plt.show()
    plt.close()
    return None

def plot_trajectrories(y, path =None, format = "png", n_trajectories=13, show=1, **kwargs):
    """
    Plot selected trajectories of 'y' insterpreted as points in R^2 or R^3, using scatter from matplotlib.
    
    'y' is assumed to be the putput of the flow applied to a batch of points. 
    The first dimension of 'y'is assmed to correspond to time, 
    the second is the batch dimension, the third corresponds to the spacial dimensions.
    If the points are in R^n for n> 3 then only the first two and the last spacial coordinates are plotted. 

    Parameters
    ----------
    y : torch.Tensor - shape (time steps, batch size, dim)
        flow.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        image format. The default is "png".
    n_trajectories : int, optional
        Number of trajectories. The default is 13.
    **kwargs : 
        Options passed to matplotlib:
            "title" - default: ""
            "xlim" - limits for the x coordinate
            "ylim" - limits for the y coordinate
            "zlim" - limits for the z coordinate

    Returns
    -------
    None.

    """
    try: title = kwargs["title"]
    except KeyError: title =""
    
    step = y.shape[1]//n_trajectories
    y= y[:,::step,:].to(device="cpu")

    fig = plt.figure()
    
    if len(y[0,0])==2:
        ax = fig.add_subplot()
        for i in range(y.shape[1]):
            xs = y[:,i,0]
            ys = y[:,i,1]
            ax.scatter(xs, ys, marker=".")

        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$y_1$')
    else:
        ax = fig.add_subplot(projection='3d')
        for i in range(y.shape[1]):
            xs = y[:,i,0]
            ys = y[:,i,-1]
            zs = y[:,i,1]
            ax.scatter(xs, ys, zs, marker=".")
    
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$x_1$')
        ax.set_zlabel(r'$y_1$')
        
    try: 
        ax.set_xlim(*kwargs["xlim"]) 
    except KeyError:
        pass
    try: 
        ax.set_ylim(*kwargs["ylim"]) 
    except KeyError:
        pass
    try:
        ax.set_zlim(*kwargs["zlim"]) 
    except KeyError:
        pass

    ax.set_title(title)
    if path != None: plt.savefig(path, format, pad_inches=0.35, bbox_inches='tight')
    if show: plt.show()
    plt.close() 
    return None


def plot_loss(loss_list, path=None, format = "png", show=1, **kwargs):
    """
    Plot the elements af 'loss_list' with log scale on y axis.

    Parameters
    ----------
    loss_list : list of tensors or dicts
        List of Losses.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        image format. The default is "png".
    no_show : bool, optional
        Switch to display the plot. The default is 0.
    **kwargs : 
        Options passed to matplotlib:
            labels
            title
            xlabel
            ylabel
            

    Returns
    -------
    None.

    """    
    try: labels = kwargs["labels"]
    except KeyError: labels = [str(i) for i in range(len(loss_list))]
    
    fig = plt.figure()
    ax = fig.add_subplot()
    for i in range(len(loss_list)):
        if type(loss_list[i])== dict:
            X= sorted(list(loss_list[i].keys()))
            Y= [loss_list[i][x] for x in X]
        else:
            X= range(len(loss_list[i]))
            Y= loss_list[i]
        ax.plot(X, Y,  marker=",", label =labels[i])
    
    ax.set_yscale('log')
    try: ax.set_title(kwargs["title"])
    except KeyError: pass    
    try: ax.set_xlabel(kwargs["xlabel"])
    except KeyError: pass
    try: ax.set_ylabel(kwargs["ylabel"])
    except KeyError: pass
    
    ax.legend()
    
    if path != None: plt.savefig(path, format = format)
    if show: plt.show() 
    plt.close()    
    return None


def plot_nice(y_pred, X, path =None, format = "png",  **kwargs):
    """
    Plot the data, trajectories and the polyfold structure together.

    Parameters
    ----------
    y_pred : torch.Tensor - shape (time steps, batch size, dim)
        Flow.
    X : 
        Vector field on the polyfold. Usd to extrct the polyfold information.
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        Image format. The default is "png".
    **kwargs : 
        n_trajectories - number of trajectories
        Options passed to matplotlib:
            title
            
    Returns
    -------
    None.

    """
    try: title = kwargs["title"]
    except KeyError: title =""
    
    try: n_trajectories = kwargs["n_trajectories"]
    except KeyError: n_trajectories = 5

    y_pred=y_pred.detach().to(device="cpu")
    dim_ambient = len(y_pred[0,0])
    colors = ['#1f77b4', '#ff7f0e'] # default blue and orange
    

    fig = plt.figure()  
    # plot start data      
    if dim_ambient==2:
        ax = fig.add_subplot()
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$y_1$')

        xs = y_pred[0,:,0]
        ys = y_pred[0,:,1]
        ax.scatter(xs, ys, marker=".", c =colors[0])
    else:
        ax = fig.add_subplot(projection='3d')
        ax.set_xlabel(r'$\tau$')
        ax.set_ylabel(r'$x_1$')
        ax.set_zlabel(r'$y_1$')
        
        xs = y_pred[0,:,0]
        ys = y_pred[0,:,-1]
        zs = y_pred[0,:,1]
        ax.scatter(xs, ys, zs, marker=".", c =colors[0])
        
    # plot final configuration
    if dim_ambient==2:
        xs = y_pred[-1,:,0]
        ys = y_pred[-1,:,1]
        ax.scatter(xs, ys, marker=".", c =colors[1])
    else:      
        xs = y_pred[-1,:,0]
        ys = y_pred[-1,:,-1]
        zs = y_pred[-1,:,1]
        ax.scatter(xs, ys, zs, marker=".", c =colors[1])
    
    # plot plane
    xs= torch.tensor([X.x1_compressed, X.x1_decompressed], dtype=torch.float32).to(device="cpu")
    
    if dim_ambient==2:
        ys=[0]
        ax.plot(xs, ys, marker="._",  alpha=0.3)
    else:
        y_start = min(y_pred[:,:,-1].flatten())  - 0.5
        y_end = max(y_pred[:,:,-1].flatten()) + 0.5
        ys=torch.tensor([y_start, y_end], dtype=torch.float32)
        xs, ys = torch.meshgrid(xs, ys)
        zs=torch.zeros(xs.shape).to(device="cpu")
        ax.plot_surface(xs, ys, zs,  alpha=0.3)

    # plot selected trajectories
    step = y_pred.shape[1] //(n_trajectories-1)
    y_pred= torch.cat([y_pred[:,::step,:], y_pred[:,-1:,:]], dim=1)
    if dim_ambient==2:
        for i in range(y_pred.shape[1]):
            xs = y_pred[:,i,0]
            ys = y_pred[:,i,1]
            ax.scatter(xs, ys, marker=".", c ='#2ca02c') #  green

    else:
        for i in range(y_pred.shape[1]):
            xs = y_pred[:,i,0]
            ys = y_pred[:,i,-1]
            zs = y_pred[:,i,1]
            ax.scatter(xs, ys, zs, marker=".", c='#2ca02c') # green

    ax.set_title(title)
    if path != None: plt.savefig(path, format= format, pad_inches=0.35, bbox_inches='tight')
    plt.show()  
    plt.close
    return None


def live_report(y_pred_val, loss_train_dict, loss_validation_dict, path_report, 
                itr,  hpara, monoton_train_dict=None, monoton_validation_dict=None, plot_live=0):  
    """
    Report on the model during training. Prints the current losses, based on 'itr'. Plots and saves losses and trajectories, saves losses.

    Parameters
    ----------
    y_pred_val : torch.Tensor - shape (time steps, batch size, dim)
        DESCRIPTION.
    loss_train_dict : dict
        Dictionary containing the training loss.
    loss_validation_dict : dict
        Dictionary containing the validation loss.
    path_report : string
        Path where the report is saved.
    itr : int
        Iteration/ epoch to report on.
    hpara : dict
        Dictionary with hyperparameters of the model.
    monoton_train_dict : dict, optional
        Dictionaray containing the training monotonicity defect used for one dimensional reconstruction models. The default is None.
    monoton_validation_dict : TYPE, optional
        Dictionaray containing the validation monotonicity defect used for one dimensional reconstruction models. The default is None.

    Returns
    -------
    None.

    """
    print('Iter {:04d} | Training Loss {:.6f} | Validation Loss {:.6f}'.format(itr, loss_train_dict[itr], loss_validation_dict[itr]))
    print()
    if hpara["test"]:
        plot_loss([loss_train_dict, loss_validation_dict], title="loss", show=plot_live)
        if monoton_train_dict!= None: plot_loss([monoton_train_dict, monoton_validation_dict], title="monotonicty defect", show=plot_live)
        plot_trajectrories(y_pred_val, show = plot_live)
    else:
        plot_loss([loss_train_dict, loss_validation_dict], path=path_report/"loss.png", show=plot_live)
        with open(path_report/"loss.csv", "w", newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=loss_train_dict.keys())
            writer.writeheader()
            writer.writerow(loss_train_dict)
            writer.writerow(loss_validation_dict)
        if monoton_train_dict!= None:
            plot_loss([monoton_train_dict, monoton_validation_dict], path=path_report/"monoton.png", show=plot_live)
            with open(path_report/"monotone.csv", "w", newline='') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=monoton_train_dict.keys())
                writer.writeheader()
                writer.writerow(monoton_train_dict)
                writer.writerow(monoton_validation_dict)

    return None

def report(X, data, loss_function, hpara, path, notes=[]):
    """
    Report the most relevant component of an experiment.

    Parameters
    ----------
    X :
        Vector field model
    data : list of custom data classes
        list of data used in the experiment
    loss_function: function
        
    hpara : dict
        additional hyper parameters 
    path : str or Path
        save path
    notes : list of strings, optional
        additional notes. The default is [].

    Returns
    -------
    None.

    """
    def get_instace_parameters(k, instance):
        """
        Extract parametrization and transformations from a composit transformation.

        Parameters
        ----------
        k : string
            Attribute name / dict key. Just used as string for book keeping here.
        instance : 
            parametrization or transformation class.

        Returns
        -------
        string_list : list of strings
            List of attributes of "instance".

        """
        string_list=[]
        if type(instance).__name__ == "Composition_Trafo":
            string_list.append(str(k)+": "+type(instance).__name__ +"\n")
            string_list.append("Transformation 1")
            string_list.extend(get_instace_parameters("",instance.trafo_1))
            string_list.append("Transformation 2")
            string_list.extend(get_instace_parameters("",instance.trafo_2))
            
        else:
            string_list.append(str(k)+": "+type(instance).__name__ +"\n")
            for key in instance.__dict__.keys():
                string_list.append("\t" + str(key) +": "+  str(instance.__dict__[key])  +"\n")
                
        return string_list
         
    with open(path, "w") as f:
        f.write("Vector field configuration" +"\n")
        f.write("------------------------------" +"\n")
        f.write("name_vectorfield: " + type(X).__name__ +"\n")
        for k in X.__dict__.keys():
            if k != "training" and k[0] != "_": # filter inherited internal attributes
                f.write(str(k)+": "+str(X.__dict__[k]) +"\n")
        f.write("\n")
        

        f.write("Data configuration" +"\n")
        f.write("------------------------------" +"\n")
        for d in data:
            f.write("name_data: " + type(d).__name__ +"\n")
            for k in d.__dict__.keys():
                if k != "dataset": # dont save the actual data here
                    if k == "parameterization" or k== "transformation": # explore the attributes of the para and trafo classes
                        # check if there is a composition and get parametes interatively
                        instance = d.__dict__[k]
                        string_list = get_instace_parameters(k, instance)
                        for string in string_list:
                            f.write(string)
                            
                    else: f.write(str(k)+": "+str(d.__dict__[k]) +"\n")
            f.write("\n")
        
        
        f.write("Loss function" +"\n")
        f.write("------------------------------" +"\n")
        f.write("name_loss: " + type(loss_function).__name__ +"\n")
        for k in loss_function.__dict__.keys():
            if k != "training" and k[0] != "_": # filter inherited internal attributes for torch loss functions
                f.write(str(k)+": "+str(loss_function.__dict__[k]) +"\n")
        f.write("\n")
        
        
        f.write("Additional hyper parameters" +"\n")
        f.write("------------------------------" +"\n")        
        for k in hpara.keys():
            f.write(str(k)+": "+str(hpara[k]) +"\n")
        f.write("\n")
        
        for note in notes:
            f.write(str(note) +"\n")
    return None


def string_to_tensor(tensor_str):
    """
    Read tensors from saved strings.

    Parameters
    ----------
    tensor_str : string
        Tensor as a string.

    Returns
    -------
    tensor or string
        Data as tensor if the transformation was successful or return the original sting.

    """
    if tensor_str.strip().startswith(("tensor", "[")):
        # Remove 'tensor(' prefix and the closing ')'
        inner = tensor_str.strip()
        inner = inner.strip("tensor")
        # Remove device info like ", device='cuda:0'"
        inner = re.sub(r",\s*device='[^']*'", "", inner)
        try:
            # Safely evaluate the string to a Python list
            data = ast.literal_eval(inner)
            return torch.tensor(data)
        except (ValueError, SyntaxError):
            # If parsing fails, return the original string
            return tensor_str
    else:
        return tensor_str


def read_report(path):
    """
    Read hyperparameters from the report saved by 'report'.

    Parameters
    ----------
    path : string
        Path to read the report from.

    Returns
    -------
    hpara : dict
        Dictionary containing the hyperparameters of a model.

    """
    hpara={}
    with open(path, "r", encoding="utf-8") as file:
        for line in file:
            if ':' in line:
                key, value = line.split(':', 1)
                # dont overwrtite things, really just for the data wich is reported twice; training data first
                if key not in hpara.keys():
                    try: hpara[key.strip()] = float(value.strip())
                    except ValueError:
                        hpara[key.strip()] = string_to_tensor(value.strip())
    # ctach some types manualy
    if "basis" in hpara.keys(): 
        basis_strings =  hpara["basis"].strip("()").split("),")
        hpara["basis"] = [ string_to_tensor(string + ")") for string in basis_strings ]
        for i in range(len(hpara["basis"])):
            if type(hpara["basis"][i]) == str:
                hpara["basis"].pop(i)
    
    if type(hpara["domain"])== str : 
        strings =  hpara["domain"].split(",")
        try: 
            strings = [string.strip().strip("()") for string in strings]
            hpara["domain"] = [[float(strings[i]), float(strings[i+1])] for i in range(len(strings)//2) ]   
        except:
            print("Warning: domain could not be reconstructed!")
        
    for key in hpara.keys():
        if key != "sample_size":
            try:
                if hpara[key] == int(hpara[key]):
                    hpara[key] = int(hpara[key])
            except (ValueError, TypeError):
                pass
    hpara["use_ode_adjoint"] = bool(hpara["use_ode_adjoint"]) 
    return hpara