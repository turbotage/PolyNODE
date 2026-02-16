import torch
import math
from pathlib import Path

from polynode import util_lib as util
from polynode import model_lib as model
from polynode import data_lib 
from polynode import loss_lib



def generate_data(hpara):
    """
    Generate training and validation data from the hyperparameters in 'hpara'.
    
    Asmumes the transformation is 'Shift' and the parametrization is "Sphere_nd"

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
        Validation dataset, sampled on a grid.
        
    """
    # time        
    if hpara["ode_method"] in ['euler', 'midpoint', "heun2", "heun3",'rk4']: 
        t=torch.linspace(0., hpara["t_end"], hpara["time_steps"] )
    else: t = torch.tensor([0., hpara["t_end"]/2, hpara["t_end"]])   
    
    # transformation       
    trafo=  data_lib.Shift(hpara["x_shift"]) 
    
    # parametrisation
    # sphere
    phi_para = data_lib.SphereND(x0=hpara["x0"], r=hpara["r"], basis= hpara["basis"])
    hpara["domain"] = torch.tensor([[0, 2*math.pi], [-math.pi/2, math.pi/2]])
    sample_method_val= "uniform" # "random"
    sample_size_val = [40, 20] 
    
    # genrate data 
    data_train = data_lib.DatasetSynth(phi_para, hpara["domain"], trafo, note="training")
    dataset_train = data_train.sample(hpara["sample_size"], method=hpara["sample_mode"])
 
    data_val = data_lib.DatasetSynth(phi_para, hpara["domain"], trafo, note="validation")
    dataset_val = data_val.sample(sample_size_val, method=sample_method_val)
    
    return (t, dataset_train, dataset_val)


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


save = 0 # switch to save the evaluation results

model_name= "checkpoint.pt" 
run_name= "SphereND_CompVFReduced_poly_structure_4_3"
path_root = Path(__file__).resolve().parents[2]
path_report = path_root/"saves"/"reconstruction"/run_name
path_save = path_root/"evaluation"/"reconstruction"/run_name
Path(path_save).mkdir(parents=True, exist_ok=True)
save_format = "png" # "pdf" #

# read hparameters from report
hpara= util.read_report(path_report/"config.txt")
hpara["device"] = torch.device('cuda:' + str(hpara["gpu"]) if torch.cuda.is_available() else 'cpu')


with hpara["device"]:
    # setup neural network part of the vectorfield
    X = model.CompVFReduced(width=hpara["width"], dim_comp=hpara["dim_comp"], dim_free=hpara["dim_free"]) 
    X.x1_free = hpara["x1_free"]
    X.x1_free_buffer = hpara["x1_free_buffer"]
    X.x1_decompressed=hpara["x1_decompressed"]
    X.x1_buffer_decompress = hpara["x1_buffer_decompress"]
    try: X.X1_speed=hpara["X1_speed"]
    except KeyError():pass
    X.load_state_dict(torch.load(path_report/model_name, weights_only=True, 
                                 map_location=torch.device(hpara["device"])))
    X.eval()
    flow = model.FlowModel(X, hpara)
    
    loss_function = loss_lib.LossLatent(hpara["x_min"], hpara["x_max"], weight= hpara["weight"])
    
    # data setup; re draw the data instead of loading it
    (t, dataset_train, dataset_val) = generate_data(hpara)   
    
    # evaluate the model
    y0_train, y_target_train = dataset_train[:] 
    y0_val , y_target_val =dataset_val[:]
    with torch.no_grad():
        y_pred_train = flow(t, y0_train)
        y_pred_val = flow(t, y0_val) 
    
    y_latent_val = y_pred_val[len(t)//2]
    # calculate  errors
    diam_data = max( torch.linalg.vector_norm(p - q) for p in dataset_val[:][0]  for q in dataset_val[:][0]).item()
    max_error, mean_error =  reconstruction_defect(y_pred_val[-1], y_target_val)
    
    print("Reconstruction max error: ", round(max_error, 4), " or ", round(max_error/diam_data*100,2), "% of diameter")
    print("Reconstruction mean error: ", round(mean_error,4), " or ", round(mean_error/diam_data*100,2), "% of diameter")
    comp_defct_max, comp_defect_var = compression_defect(y_latent_val, hpara["dim_comp"])
    print("Maximum compression defect: ", comp_defct_max)
    print("Deviation compression defect: ", comp_defect_var)
            
    #plot encoding and decoding, assuming len(t) = 500, and trajectories
    if save:
        i_snapshot = [0, 79, 142, 250] #torch.linspace(0, len(t)//2, 5, dtype=int)
        util.plot_data_4d([y_pred_val[i] for i in i_snapshot], use_color_map = True, 
                          path= path_save / "snapshots_1.{}".format(save_format), format=save_format)
        
        i_snapshot_re = [310, 380, 498] #torch.linspace(len(t)//2, len(t)-1, 5, dtype=int)
        util.plot_data_4d([y_pred_val[i] for i in i_snapshot_re], use_color_map = True, 
                          path= path_save / "snapshots_2.{}".format(save_format), format=save_format)
        util.plot_trajectrories_4d(y_pred_val, 
                                   path=path_save/"trajectories.{}".format(save_format), format=save_format)
        
    else:
        i_snapshot = [0, 79, 142, 250] #torch.linspace(0, len(t)//2 , 5, dtype=int)
        util.plot_data_4d([y_pred_val[i] for i in i_snapshot], titel = "snapshots_1", use_color_map = True)
        
        i_snapshot_re = [310, 380, 498] #torch.linspace(0, len(t)//2 , 5, dtype=int)
        util.plot_data_4d([y_pred_val[i] for i in i_snapshot_re], titel = "snapshots_2", use_color_map = True)

        util.plot_data_4d([y_latent_val], title="latent state, validation")
        util.plot_trajectrories_4d(y_pred_val)
        
        y_latent_val_3d = torch.cat( [y_latent_val[:,:1], y_latent_val[:,2:]], dim=1 )
        util.plot_data([y_latent_val_3d] )