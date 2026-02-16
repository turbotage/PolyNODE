import time
import torch
from pathlib import Path
import math
import matplotlib.pyplot as plt
import csv

from polynode import util_lib as util
from polynode import model_lib as model
from polynode import data_lib 
from polynode import loss_lib

def get_validation_size(target, train_size):
    """
    Find sampling stride for valodation data such that training data and validation data are disjoint.

    Parameters
    ----------
    target : TYPE
        DESCRIPTION.
    train_size : TYPE
        DESCRIPTION.

    Returns
    -------
    i : TYPE
        DESCRIPTION.

    """
    for i in range(int(target), int(target*1.3)):
        gdc = math.gcd(i, train_size)
        if gdc==1:
            return i
    print("No validation scize with gdc 1 found, choose new 'target' or 'train_size'.")
    return None
    

get_validation_size(270, 900)

def generate_data_spiral(hpara):
    """
    Generate training and validation data from the hyperparameters in 'hpara'.
    
    Asmumes the transformation is 'Shift' and the spiral is parametrization is "Spiral_nd"

    Parameters
    ----------
    hpara : dict
        Dictionary containing the hyperparameters of an experiment.

    Returns
    -------
    t : TYPE
        Time array.
    dataset_train : TYPE
        Training Dataset.
    dataset_val_odered : TYPE
        Validation dataset order by sampling in the domain of the parametrisation.
    dataset_val_rando : TYPE
        Validation dataset, randomly sampled.
    length_intrisic : float
        Intrinsinc length of the curve between the start and end of the domain.
    dist : TYPE
        Intrinsic distance function of the one dimensional data manifold.

    """
    # time        
    if hpara["ode_method"] in ['euler', 'midpoint', "heun2", "heun3",'rk4']: 
        t=torch.linspace(0., hpara["t_end"], hpara["time_steps"] )
    else: t = torch.tensor([0., hpara["t_end"]/2, hpara["t_end"]])   
    
    # transformation       
    trafo = data_lib.Shift(hpara["x_shift"]) 
    
    # parametrisation
    # spiral
    phi_para = data_lib.SpiralND(hpara["x0"], hpara["speed_radius"],speed_angle=hpara["speed_angle"] , 
                                 phi_0= hpara["phi_0"], r_min=hpara["r_min"], basis= hpara["basis"])
    dist = data_lib.DistanceSpiral(phi_para)
    y_temp= phi_para(hpara["domain"]  [0])
    length_intrisic = (dist(y_temp)[1] - dist(y_temp)[0]).item()

    sample_size_val = [  get_validation_size(int(hpara["sample_size"][0]*0.3), int(hpara["sample_size"][0])) ]
    hpara["sample_size_val"] = sample_size_val  
    print("sample size, train:", hpara["sample_size"])
    print("sample size, val:", sample_size_val)
     
    # genrate data
    data_train = data_lib.DatasetSynth(phi_para, hpara["domain"]  , trafo, note="training")
    dataset_train = data_train.sample(hpara["sample_size"], method=hpara["sample_mode"], dist=dist) 
  
    data_val = data_lib.DatasetSynth(phi_para, hpara["domain"]  , trafo, note="validation")
    dataset_val_odered = data_val.sample(sample_size_val, method=hpara["sample_mode"], dist=dist)

    dataset_val_rando = data_val.sample(sample_size_val, method="random")
    
    return (t, dataset_train, dataset_val_odered, dataset_val_rando, length_intrisic, dist)


def compression_defect(y_latent, dim_comp):
    """
    Calculate the maxmimum error compared to 0 and the maximal deviation of the compresed compoments.

    Parameters
    ----------
    y_latent : TYPE
        Flow at latent time.
    dim_comp : TYPE
        number of compresed dimensions.

    Returns
    -------
    norm_max : TYPE
        Maximum error.
    variance : TYPE
        Maximal deviation.

    """
    y_comp = y_latent[:, 1:1+dim_comp]
    norm_max = max(torch.linalg.vector_norm(y_comp, dim=1))
    deviation = max(y_comp.flatten()) - min(y_comp.flatten())
    return  norm_max, deviation

def reconstruction_defect(y_pred, y_target):
    """
    Calculate the maximum and mean eucleadien reconstruction error of the flow.

    Parameters
    ----------
    y_pred : TYPE
        Flow at end time.
    y_target : TYPE
        Reconstruction target.

    Returns
    -------
    max_error : TYPE
    mean_error : TYPE

    """
    error = torch.linalg.vector_norm(y_pred - y_target, dim=1)
    max_error = max(error).item()
    mean_error = sum(error).item()/len(error)
    return  max_error, mean_error


def check_monotonicity(y):
    """
    Check the monotonicity of the one dimensional encoding.

    Parameters
    ----------
    y : tensor
        Time slice of the flow.

    Returns
    -------
    TYPE
        Number of points out of oder.

    """
    orientation = torch.sign(y[-1, -1] - y[0, -1])
    defect = torch.sign(y[1: , -1] -  y[:-1,-1])
    return torch.sum( torch.abs((defect - orientation)/2) ).item()

def monotonicity_position(y):
    """
    Find the posions where monotonicity fails.

    Parameters
    ----------
    y : TYPE
        Time slice of a flow.

    Returns
    -------
    positions : list
        List of positions where monotonictity fails.

    """
    orientation = torch.sign(y[-1, -1] - y[0, -1])
    defect = torch.sign(y[1: , -1] -  y[:-1,-1])
    positions=[]
    for i in range(len(defect)):
        if defect[i] != orientation:
            positions.append(i)
    return positions


def plot_errors(err, labels, title ="", path=None, format="png", y_label='err'):
    """
    Plot errors for several experiments.

    Parameters
    ----------
    err : dict
        Dictionary of errors. Elements are tuples.
    labels :  
        Labels for the plot.
    title : TYPE, optional
        Title for the plot. The default is "".
    path : string, optional
        Path where the plot is saved. The default is None.
    format : sting, optional
        Image format. The default is "png".
    y_label : TYPE, optional
        y lable of the plot. The default is 'err'.

    Returns
    -------
    None.

    """
    fig = plt.figure()  
    ax = fig.add_subplot()
    keys = sorted(err.keys())
    for j in range(2):
        ys = [err[k][j] for k in keys]
        ax.scatter(keys, ys, marker=".", label= labels[j])
    ax.legend()
    ax.set_xlabel('Speed')
    ax.set_ylabel(y_label)
    ax.set_title(title)
    if path != None: plt.savefig(path, format = format)
    plt.show()
    
    return None




save = 0 # switch to save the evaluation results

low_resolution = 1 # switch to halve the sample size


# error dictionaries
final_loss = {}
err_reconstr = {}
err_reconstr_rel = {}
err_mono = {}
err_mono_rel = {}

# index set to loop over through the experiments
index_set = range(1,11)
for index in index_set:
    speed= index/2
    print("\n", "evaluation of index: ", index, "\n")
    
    model_name= "checkpoint.pt"  
    run_name= "SpiralND_CompVFReduced_poly_structure_4_2_speed_{}".format(speed)
    
    path_root =  Path(__file__).resolve().parents[2]
    path_report= path_root/"saves"/"reconstruction"/run_name     
    path_save = path_root/"evaluation"/"reconstruction"/run_name
    Path(path_save).mkdir(parents=True, exist_ok=True)
    save_format = "pdf" # "png"
    
    # read hyper parameters from report
    hpara= util.read_report(path_report/"config.txt")
    hpara["device"] = torch.device('cuda:' + str(hpara["gpu"]) if torch.cuda.is_available() else 'cpu')

    if low_resolution: hpara["sample_size"]=hpara["sample_size"]//2

    with hpara["device"]:
        # setup neural network part of the vector field
        X = model.CompVFReduced(width=hpara["width"], dim_comp=hpara["dim_comp"], 
                                dim_free=hpara["dim_free"]) 
        X.x1_free = hpara["x1_free"]
        X.x1_free_buffer = hpara["x1_free_buffer"]
        try: X.X1_speed=hpara["X1_speed"]
        except KeyError():pass
        X.load_state_dict(torch.load(path_report/model_name, weights_only=True, 
                                     map_location=torch.device(hpara["device"])))
        X.eval()
        flow = model.FlowModel(X, hpara)

        # data setup; redraw the data instead of loading it
        (t, dataset_train, dataset_val_ordered, dataset_val_rando, length_intrinsic, dist) = generate_data_spiral(hpara)   

        # evaluate the model
        y0_train, y_target_train = dataset_train[:] 
        y0_val_ordered , y_target_val_ordered =dataset_val_ordered[:]
        y0_val_rando , y_target_val_rando =dataset_val_rando[:]
        
        with torch.no_grad():
            y_pred_train = flow(t, y0_train)
            y_pred_val_ordered = flow(t, y0_val_ordered)
            y_pred_val_rando = flow(t, y0_val_rando)
        

        # calculate unwinding error
        mono_pre = check_monotonicity(y_pred_val_ordered[len(t)//4])
        mono_latent = check_monotonicity(y_pred_val_ordered[len(t)//2])
        print("monotonicity defect at T/4 ", mono_pre)
        print("monotonicity defect, validation at T/2: ", mono_latent)
        
        err_mono[speed] = ( check_monotonicity(y_pred_train[len(t)//2]) , mono_latent)
        err_mono_rel[speed] = ( err_mono[speed][0]/y0_train.shape[0] *100 , mono_latent/y0_val_ordered.shape[0]*100)
        
        if mono_pre >0:
            print("defect positions: ", monotonicity_position(y_pred_train[len(t)//4]))
        if mono_latent >0:
            print("defect positions: ", monotonicity_position(y_pred_train[len(t)//2]))
        
        
        # calculate final losses
        # shuffle the training data
        loader_train =  torch.utils.data.DataLoader(dataset_train, batch_size=hpara["batch_size"], 
                            shuffle=True , num_workers = hpara["num_workers"], generator=torch.Generator(device=hpara["device"]))
        
        loss_function = loss_lib.LossPreprocessIsometry(X.x1_compressed, X.x1_decompressed, X.dim_comp, dist, x1_pre=X.x1_free,
                                                          weight=1, iso_scale= length_intrinsic**0.5 ) 
        
        final_loss[speed] = (loss_function(y_pred_train, y_target_train, t) , 
                             loss_function(y_pred_val_rando, y_target_val_rando, t) ) 
        
        # calculate reconstruction error
        diam_data = max( torch.linalg.vector_norm(p - q) for p in dataset_val_rando[:][0]  for q in dataset_val_rando[:][0]).item()
        max_error, mean_error =  reconstruction_defect(y_pred_val_rando[-1], y_target_val_rando)
        
        print("reconstruction max error: ", round(max_error, 4), " or ", round(max_error/diam_data*100,2), "% of diameter")
        print("reconstruction mean error: ", round(mean_error,4), " or ", round(mean_error/diam_data*100,2), "% of diameter")
        
        err_reconstr[speed] = (reconstruction_defect(y_pred_train[-1], y_target_train)[1],mean_error)
        err_reconstr_rel[speed] = (err_reconstr[speed][0]/diam_data*100,  err_reconstr[speed][1]/diam_data*100)
        
        #plot individual runs
        if save:
            # snapshot plot unwinding
            i_snapshot = torch.linspace(0, len(t)//2 - 50, 5, dtype=int)
            util.plot_data([y_pred_val_ordered[i] for i in i_snapshot], use_color_map = True, 
                           path=path_save/"snapshot_{}.{}".format(index, save_format), format =save_format)
            util.plot_trajectrories(y_pred_val_ordered)
            util.plot_data([y_target_val_ordered, y_pred_val_ordered[-1]], 
                           labels=["target", "reconstruction"], use_color_map = True, 
                           path= path_save/"reconstruction_{}.{}".format(index, save_format), format =save_format)
            util.plot_nice(y_pred_train,X, n_trajectories=8, 
                           path=path_save/"nice_plot_{}.{}".format(index, save_format), format =save_format)

            # 2d reconstruction
            util.plot_data([y_target_val_ordered[:, 1: : 2] ,y_pred_val_ordered[-1, :, 1: : 2]], 
                           labels=["target", "reconstruction"], path=path_save/"reconstruction_2d_{}.{}".format(index, save_format), 
                           format =save_format)

            # plot monotonicity during training
            with open(path_report/'monotone.csv', newline='') as csvfile:
                mono_error_training, mono_error_val = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            
            step = 10
            mono_error_training = {step*k : mono_error_training[step*k]/hpara["sample_size"].item()*100 for k in mono_error_training.keys()  if step*k < max(mono_error_training.keys()) }
            mono_error_val = {step*k : mono_error_val[step*k]/hpara["sample_size_val"][0]*100 for k in mono_error_val.keys() if step*k < max(mono_error_training.keys()) }
            
            util.plot_loss([mono_error_training, mono_error_val], xlabel = "Epochs", ylabel=r"Relative Monotonicity Error [\%]",
                           path= path_save/"monotonicity_{}.{}".format(index, save_format), format =save_format, labels=["training", "validation"])
            
        else:
            # snapshot plot unwinding
            i_snapshot = torch.linspace(0, len(t)//2 - 50, 5, dtype=int) 
            util.plot_data([y_pred_val_ordered[i] for i in i_snapshot], 
                           title="snapshots, validation", use_color_map = True)
            util.plot_trajectrories(y_pred_val_ordered)
            util.plot_data([y_target_val_ordered, y_pred_val_ordered[-1]], 
                           title="Data reconstruction all, validation", 
                           labels=["target", "reconstruction"], use_color_map = True)
            util.plot_nice(y_pred_train,X, n_trajectories=8)

            #snapshot plot rewinding
            i_snapshot = torch.linspace(len(t)//2, len(t)-1, 5, dtype=int)
            util.plot_data([y_pred_val_ordered[i] for i in i_snapshot], 
                           title="snapshots, validation", use_color_map = True)
            
            #snapshot whole
            i_snapshot = torch.linspace(0, len(t)-1, 8, dtype=int)
            util.plot_data([y_pred_val_ordered[i] for i in i_snapshot], 
                           title="snapshots, validation", use_color_map = True)
            
            # 2d reconstruction
            util.plot_data([y_target_val_ordered[:, 1: : 2] ,y_pred_val_ordered[-1, :, 1: : 2]], title="reconstruction 2d")

            # plot monotonicity during training
            with open(path_report/'monotone.csv', newline='') as csvfile:
                mono_error_training, mono_error_val = csv.DictReader(csvfile, quoting=csv.QUOTE_NONNUMERIC)
            
            step = 10
            mono_error_training = {step*k : mono_error_training[step*k]/hpara["sample_size"].item()*100 for k in mono_error_training.keys()  if step*k <= max(mono_error_training.keys()) }
            mono_error_val = {step*k : mono_error_val[step*k]/hpara["sample_size_val"][0]*100 for k in mono_error_val.keys() if step*k <= max(mono_error_training.keys()) }
            
            util.plot_loss([mono_error_training, mono_error_val], xlabel = "Epochs", ylabel=r"Relative Monotonicity error [\%]")
            
    
# plot error over index
if save: 
    plot_errors(final_loss, labels=["training", "validation"], y_label="Loss", 
                path=path_save/"loss.{}".format(save_format), format =save_format)
    plot_errors(err_reconstr, labels=["training", "validation"], y_label="Reconstruction Error", 
                path=path_save/"recon_error.{}".format(save_format), format =save_format)
    plot_errors(err_reconstr_rel, labels=["training", "validation"], y_label=r"Relative Reconstruction Error [\%]", 
                path=path_save/"recon_err_rel.{}".format(save_format), format =save_format)
    plot_errors(err_mono, labels=["training", "validation"], y_label="Monotonicity Error", 
                path=path_save/"monotonicity_err.{}".format(save_format), format =save_format)
    plot_errors(err_mono_rel, labels=["training", "validation"], y_label=r"Relative Monotonicity Error [\%]", 
                path=path_save/"monotonicty_err_rel.{}".format(save_format), format =save_format)
else:
    plot_errors(final_loss, labels=["training", "validation"], title= "loss")
    plot_errors(err_reconstr , labels=["training", "validation"], title= "reconstruction error")
    plot_errors(err_reconstr_rel, labels=["training", "validation"], title= "reconstruction error, relatve")
    plot_errors(err_mono , labels=["training", "validation"],  title= "monotonicty error")
    plot_errors(err_mono_rel, labels=["training", "validation"], title= "monotonicty error, relative")
                

