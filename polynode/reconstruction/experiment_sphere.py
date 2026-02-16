import time
import torch
import torch.optim as optim
from pathlib import Path
import datetime

from polynode import util_lib as util
from polynode import model_lib as model
from polynode import data_lib 
from polynode import loss_lib

path_root = Path(__file__).resolve().parents[2]

# catch hyperparamter that are not part of some class in a dict, load defaults from util
hpara=util.hpara

test = 1 # switch to aktivate test mode with less dense data sampling and more verbose reporting
plot_live = 0 # switch to enable live plotting duing training
load_model=0 # switch to load a model from the path_report_load

with hpara["device"]:
    
    # data setup
    # time        
    hpara["t_end"] = 3.5 
    if test: hpara["time_steps"] = 200  # fixed time steps for non adaptive solvers
    else: hpara["time_steps"] = 500
    t=data_lib.get_time(hpara)

    # parametrisation
    # sphere
    x0= torch.tensor([-7.,3.,0., 0.])     
    e1 = torch.tensor([1., 0., 0.,  0.])
    e2 = torch.tensor([0., 0., 0.,  1.])
    e3 = torch.tensor([0., 1., 0.,  0.])
    e2 = e2/torch.linalg.vector_norm(e2)
    basis = (e1, e2, e3 )
    radius = 1
    phi_para = data_lib.SphereND(x0, radius, basis)
    bounds = torch.tensor( ((0.0, 2*torch.pi), (-torch.pi/2, torch.pi/2)) )   
     
    # transformation       
    #shift
    x1_shift = torch.tensor(( (-x0[0] + radius + 0.5)*2  , 0.,  0., 0.))
    shift= data_lib.Shift(x1_shift)
    trafo= shift
    
    # setup sampling
    if test:
        sample_method_train= "uniform"
        sample_size_train =   [25, 20] 
        sample_method_val= "uniform"
        sample_size_val =  [ 17 , 13] 
    else:
        sample_method_train= "uniform"
        sample_size_train =   [90, 55 ]
        sample_method_val= "random"
        sample_size_val = [50, 30]
    
    # genrate data
    data_train = data_lib.DatasetSynth(phi_para, bounds, trafo, note="training")
    dataset_train = data_train.sample(sample_size_train, sample_method_train) 
   
    data_val = data_lib.DatasetSynth(phi_para, bounds, trafo, note="validation")
    dataset_val = data_val.sample(sample_size_val, sample_method_val)
    y0_val, y_target_val = dataset_val[:]

    # note: the DataLoader is an itterator for the dataset
    loader_train =  torch.utils.data.DataLoader(dataset_train, batch_size=hpara["batch_size"], 
                        shuffle=True , num_workers = hpara["num_workers"], generator=torch.Generator(device=hpara["device"])) 

    # setup the vector field
    if not load_model:
        notes=[]
        width = 200
        dim_free = 2
        dim_comp = 1
        X = model.CompVFReduced(width=width, dim_comp=dim_comp, dim_free=dim_free)
        X.x1_decompressed = X.x1_compressed +2*radius + 1
        X.x1_buffer_decompress = X.x1_decompressed +1
        X.X1_speed=x1_shift[0].item()/hpara["t_end"]

    if load_model:
        model_name_load = "checkpoint.pt"
        run_name_load = "SphereND_CompVFReduced_poly_structure_4_3"
        path_report = path_root/"saves"/"reconstruction"/run_name_load
        hpara= util.read_report(path_report/"config.txt")
        # overwrite device for new experiment
        hpara["device"] = torch.device('cuda:' + str(hpara["gpu"]) if torch.cuda.is_available() else 'cpu')
        notes=["loaded from " + run_name_load + " , model: " + model_name_load]

        # setup neural network part of the vector field
        dim_comp=hpara["dim_comp"]
        dim_free=hpara["dim_free"]
        X = model.CompVFReduced(width=hpara["width"], dim_comp=hpara["dim_comp"], dim_free=hpara["dim_free"]) 
        X.x1_free = hpara["x1_free"]
        X.x1_free_buffer = hpara["x1_free_buffer"]
        X.X1_speed=hpara["X1_speed"]
        # try to load the constant X1 component for the CompVFReduced, will fail savely for CompVF
        try: X.X1_speed=hpara["X1_speed"]
        except KeyError():print("Could not load X1_speed from hpara. Loading weights.")
        X.load_state_dict(torch.load(path_report/model_name_load, weights_only=True, map_location=torch.device(hpara["device"])))
        
    # set test state after potentially loading hpara
    hpara["test"] = test
    
    # setup flow model with solver, loss, optimizer, scheduler, stopper
    hpara["ode_method"] = "euler"  
    hpara["rtol"] = 1e-4
    hpara["atol"] = 1e-5
    hpara["max_epochs"] = 30000     
    hpara["start_lr"] = 1e-3
    hpara["lr_end_factor"] = 0.1
    hpara["lr_end_epochs"] =  5000 
    gamma = hpara["lr_end_factor"]**(1/hpara["lr_end_epochs"])
    if test:
        hpara["test_freq"]= 5
        hpara["patience"] = 250
    else: 
        hpara["test_freq"]= 10    
        hpara["patience"] = 300
    use_amp = hpara["device"] == "cuda"
    
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
    flow = model.FlowModel(X, hpara)
    optimizer = optim.RMSprop(flow.parameters(), lr=hpara["start_lr"], momentum=hpara["momentum"])  
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma)
    early_stopper = util.EarlyStopper(patience=hpara["patience"])
    loss_function =   loss_lib.LossLatent(X.x1_compressed,  X.x1_decompressed)

    # initial reporting
    if test:
        print("Warning: running in test mode! \n")
        run_name = "test"
        path_report=None
        util.plot_data([dataset_train[:][0]], title="Training Data")
        util.plot_data([dataset_train[:][1]], title="Training Data, target")
    else:
        model_name=  "vector_field.pt" 
        now = datetime.datetime.now()
        date=str(now.day)+"."+str(now.month)+";"+str(now.hour)+"."+str(round(now.minute,-1))
        run_name = (type(phi_para).__name__ + "_" +type(X).__name__ +  "_poly_structure_"+str(dim_comp + dim_free + 1)+ "_" 
                    + str(dim_free +1)+"_"+date )
        path_report = path_root/"saves"/"reconstruction"/run_name
        path_report.mkdir(parents=True, exist_ok=True)
        data_val.save(path_report/"data_val.pt")
        util.report(X, [data_train, data_val], loss_function, hpara, path_report/"config.txt", notes=notes)
        util.plot_data([dataset_train[:][0]], title="Training Data, input", path=path_report/"training_data.png")
        util.plot_data([dataset_train[:][1]], title="Training Data, target", path=path_report/"target_data.png")
    

    # dicts for loss reporting and plotting 
    loss_validation_dict = {}
    loss_train_dict = {}
    print("Start training, run: "+run_name)
    print("----------------------------------------------------------")
    # keep track of best loss for simple checkpointing
    best_loss = float("inf")
    if test: print("start training")
    for itr in range(1, hpara["max_epochs"] + 1):  # epoch loop
        for y0_train, y_target_train in loader_train: # batch loop
            with torch.autocast(device_type=str(hpara["device"]), dtype=torch.float16, enabled=use_amp):
                y_pred_train = flow(t, y0_train)
                loss = loss_function(y_pred_train, y_target_train, t)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            lr_scheduler.step()
    
        # live reporting
        if itr % hpara["test_freq"] == 0:
            with torch.no_grad():                
                y_pred_val = flow(t, y0_val)
                loss_validation_dict[itr] =  loss_function(y_pred_val, y_target_val, t).item()
                loss_train_dict[itr] =  loss.item()
            
            util.live_report(y_pred_val, loss_train_dict, loss_validation_dict, path_report, itr, hpara, plot_live=plot_live)
            if not test:
                # save checkpoints if validation loss is better
                if loss_validation_dict[itr]<best_loss:
                    torch.save(X.state_dict(), path_report/"checkpoint.pt")
                    best_loss = loss_validation_dict[itr]
                
            if early_stopper(loss_validation_dict[itr]):             
                break  
            
# reporting after training
with torch.no_grad(): # stop tracking gradinent for evaluation
    y_pred_train = flow(t, y0_train)
    y_pred_val = flow(t, y0_val)
    
if test:
    # plot letent representation, trajectories and reconstruction
    util.plot_data([y_pred_val[0], y_pred_val[int(len(t)/2)]], title="latent state, validation", use_color_map = True)
    util.plot_trajectrories(y_pred_val)
    util.plot_data([y_target_val, y_pred_val[-1]], 
                   title="Data reconstruction, validation", labels=["target", "projected"], use_color_map = True)

else:
    # save model
    torch.save(X.state_dict(), path_report/model_name)
    
    # plot data and evaluation restults for training and validation
    util.plot_data([y_target_train, y_pred_train[-1]], title="Data reconstruction, Training", path=path_report/"reconstruction_trainig.png")
    util.plot_data([y_target_val, y_pred_val[-1]], title="Data reconstruction, Validation", path=path_report/"reconstruction_validation.png")
    
    # plot trajectories from validation dataset
    util.plot_trajectrories(y_pred_val, path=path_report/"trajectories.png")
    util.plot_nice(y_pred_train,X, n_trajectories=8 , path=path_report/"nice_plot.png")
    
    #plot loss
    util.plot_loss([loss_train_dict, loss_validation_dict], path=path_report/"loss.png")