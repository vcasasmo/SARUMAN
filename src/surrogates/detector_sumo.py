
import torch
# import gpytorch
# from botorch.models import SingleTaskGP
# from botorch.models.transforms import Normalize
# from botorch.models.transforms.outcome import Standardize
import numpy as np
from tqdm import tqdm
# import SimpleITK as sitk 

def change_coords(mesh_x, mesh_y, mesh_z, det_coords_RS1, device):
    #det_coords_RS1 = row
    x = torch.tensor(det_coords_RS1[0], device=device)
    y = torch.tensor(det_coords_RS1[1], device=device)
    z = torch.tensor(det_coords_RS1[2], device=device)
    #print(f'x: original: {x.item()}, {z.item()}')
    # Compute the angle for the rotation matrix
    theta =torch.atan2(x,-z) + torch.pi
    #print(x.cpu().numpy(),z.cpu().numpy(), theta.cpu().numpy()/np.pi * 180)
    # Rotation matrix around Y-axis 
    R = torch.tensor([[torch.cos(theta), 0, torch.sin(theta)],
                  [0, 1, 0],
                  [-torch.sin(theta), 0, torch.cos(theta)]], device=device)

    # Rotate using the matrix multiplication
    x_det = R[0,0] * x +  R[0,2] * z
    
    y_det =  y
    z_det =  R[2,0] * x +  R[2,2] * z
   
    
 
    x_prime = x_det -  R[0,0] * mesh_x - R[0,2] * mesh_z  
    
    y_prime =  -(y_det - mesh_y)
    z_prime = z_det -  R[2,0] * mesh_x - R[2,2] * mesh_z
    #print(x_prime.shape)
    #print(f'x: prime: {x_prime[15,0,15].item()}, {z_prime[15,0,15].item()}')
    return x_prime, y_prime, z_prime

def estimate_detector_efficiencies(waste_batch):
    from src.config import load_config
    config = getattr(waste_batch, "config", None) or load_config()
    nn_config = config["detector_surrogate"]["nn"]
    device = config["detector_surrogate"]["device"]
    if device != "cpu":
        raise ValueError("The production detector NN currently supports device='cpu' only")
    import pickle
    import tensorflow as tf
 
    tf.config.set_visible_devices([], 'GPU')  # Hides all GPUs from TensorFlow

    model = tf.keras.models.load_model(str(nn_config["model_path"]))
    with open(nn_config["input_scaler"],"rb") as f:
        input_scaler = pickle.load(f)
    with open(nn_config["output_scaler"],"rb") as f:
        output_scaler = pickle.load(f)

    mesh_x = torch.tensor(waste_batch.mesh.coords['x'].values, device=device,dtype=torch.float32).flatten()
    mesh_y = torch.tensor(waste_batch.mesh.coords['y'].values, device=device,dtype=torch.float32).flatten()
    mesh_z = torch.tensor(waste_batch.mesh.coords['z'].values, device=device,dtype=torch.float32).flatten()
    mesh_size = waste_batch.mesh.shape
    energies_detected = torch.tensor(waste_batch.energies_detected, device=device,dtype=torch.float32) 
    mesh_x,mesh_y,mesh_z = torch.meshgrid(mesh_x,mesh_y,mesh_z, indexing='ij')
       
    for detector_PoV in tqdm(range(waste_batch.detector_positions.height), desc="Estimating detector efficiencies"):
        row = waste_batch.detector_positions.row(detector_PoV)
        det_x, det_y, det_z = row  # tuple con todos los valores de la fila

       
        alpha_x, alpha_y, alpha_z = change_coords(mesh_x, mesh_y, mesh_z, row, device=device)
        x1 =   np.sqrt(alpha_x**2 + alpha_y**2 + alpha_z**2)
        x2 =   np.arctan2(alpha_z, alpha_x)
        x3 =   np.arccos(alpha_y / x1)
        x1,x2,x3=x1.flatten(),x2.flatten(),x3.flatten()
        #print(f'x1 shape {x1.shape}')
        #print(f"Distance for Detector {detector_PoV}:   {x1.min()}")
        for E, energy in enumerate(energies_detected):
            # Stack the input tensor for the GP model
            energy = np.sqrt(energy) 
            input_tensor = tf.stack([x1,x2,x3,energy.expand_as(x1)],axis = 1)
            input_tensor = input_scaler.transform(input_tensor)
        
       
            #waste_batch.debug_input[detector_PoV,:,:]=np.array(input_tensor)
    
           
            mean_scaled = model.predict_on_batch(input_tensor).reshape(-1,1)
            mean = output_scaler.inverse_transform(mean_scaled)
            mean=np.expm1(mean)**3
            #print(mean)
            # Reshape the posterior mean and variance back to the mesh shape
            #mean,_= gp.outcome_transform.untransform(posterior.mean)
            variance = 1
            mean = mean.reshape(mesh_size)
        
            #print(energy)
            #print(mean.min())
            
           
            
            # Assign the posterior mean and variance to the waste_batch detector efficiencies
             
            waste_batch.detector_efficiencies[detector_PoV, :, :, :, E, 0] = mean#np.transpose(mean, (2, 1, 0))#[:, ::-1, :]
            waste_batch.detector_efficiencies[detector_PoV, :, :, :, E, 1] = variance#torch.permute(variance, (1, 0, 2))
            
