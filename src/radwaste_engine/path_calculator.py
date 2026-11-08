# -*- coding: utf-8 -*-
"""
Path calculator based on rectilineous particle trajectory tracking

Requires:
mesh: an xarray entity wielding coordinates, spacing and origin of the ortogonal mesh is going to work with.
detector_rel_coords: a list with positions to be checked 
mesh_folder_path: a path to store the paths for each detector position
unique_ids: the name of each detector position file.
chunk_size = max of chunk size for parallel computation 


First, we go single mode and we only take into account the center of the voxel. In the future the points could be (in fact, they can) be
located anywhere within the voxel of origin.


In the future, GPU capabilities look interesting, I can try to use torch and/or CuPy

Created on Wed Sep 18  2024

@author: vcmolina
"""
"""
1st. The core, a simple machine that takes a point, and tracks the voxel crossed until the evaluated position leaves the mesh boundaries.
"""
import numpy as np
import polars as pl
import time
from tqdm import tqdm
import multiprocessing as mp
import os 
import gc
import concurrent.futures
_tracker = None 
def init_tracker(tracker):
    """ Initialize the global tracker variable in each worker. """
    global _tracker
    _tracker = tracker
class ParticleTracker:
    def __init__(self, voxel_size, mesh_min, mesh_max):
        self.voxel_size = np.array(voxel_size)  # Size of each voxel (dx, dy, dz)
        self.mesh_min = np.array(mesh_min)  # Minimum coordinates of the mesh
        self.mesh_max = np.array(mesh_max)  # Maximum coordinates of the mesh
        
     
    def track_particle(self, origin, destination, debug=False):
     # Initialize particle tracking information
     origin = np.array(origin)
     destination = np.array(destination)
     
     # Calculate the direction as the normalized vector from origin to destination
     direction = destination - origin
     ray_len=np.linalg.norm(direction)
     direction[direction==0]=1e-12 #jitter to avoid divide by zero on parallel axes 
     #print(direction)
     # Find the initial voxel the particle is in
     current_voxel = np.floor((origin - self.mesh_min) / self.voxel_size).astype(int)
     
     # Determine step direction (+1 or -1) based on particle direction
     step = np.sign(direction).astype(int)
     
     # Calculate t_max (distance to next voxel boundary in each axis)
     t_max = ((self.voxel_size * (current_voxel + (step > 0)) + self.mesh_min - origin) / direction)
     
     # Calculate t_delta (distance across one voxel in each direction)
     t_delta = np.abs(self.voxel_size / direction)
     
     # Track distances and voxels traversed
     total_distance_t = 0
     voxel_distances = []
     
     # Distance to first voxel boundary
     distance_in_first_voxel_t = np.min(t_max)  # Shortest distance to leave the first voxel
     
     # Include the initial voxel and the distance traversed inside it
     voxel_distances.append((tuple(current_voxel), distance_in_first_voxel_t*ray_len))
     total_distance_t += distance_in_first_voxel_t
     
     # Update t_max for the next voxel boundary crossing
     next_axis = np.argmin(t_max)
     t_max[next_axis] += t_delta[next_axis]
     current_voxel[next_axis] += step[next_axis]
     
     # Now continue tracking for the remaining voxels
     while np.all(current_voxel >=0) and np.all(current_voxel <= ((self.mesh_max - self.mesh_min) / self.voxel_size)-1):
  
         # Find the next axis to cross (smallest t_max)
         next_axis = np.argmin(t_max)
         
         # Distance to cross to the next voxel
         distance_t = t_max[next_axis] - total_distance_t
         #print(distance)
         total_distance_t += distance_t
         #print(current_voxel)
         #print(np.all(current_voxel <= ((self.mesh_max - self.mesh_min) / self.voxel_size)))

         
        #if np.any(next_voxel < 0) or np.any(next_voxel > ((self.mesh_max - self.mesh_min) / self.voxel_size)):
        #   break  # Exit the loop if the next voxel is out of bounds
         
         # Record the voxel and distance traversed
         voxel_distances.append((tuple(current_voxel), distance_t*ray_len))
         # Check if the next voxel is within the mesh dimensions
         next_voxel = current_voxel.copy()
         next_voxel[next_axis] += step[next_axis]
         current_voxel = next_voxel
         
         # Update t_max for the next voxel boundary crossing
         t_max[next_axis] += t_delta[next_axis]
     
     if debug:
         final_point = destination
         self.check_final_point(final_point, destination, voxel_distances, origin)
     
     return voxel_distances    

    def check_final_point(self, final_point, destination, voxel_distances, origin):
        """
        Check final point is for debugging purposes, it checks sum of all paths equals the euclidean distance
        """
        # Check if the final point is within the mesh bounds
        within_bounds = np.all(final_point >= self.mesh_min) and np.all(final_point <= self.mesh_max)
        
        # Calculate total voxel distances and distance from final point to destination
        total_voxel_distance = sum([d for _, d in voxel_distances])
        final_to_destination_distance = np.linalg.norm(destination - final_point)
        
        # Calculate total distance from origin to destination
        total_direct_distance = np.linalg.norm(destination - origin)
        
        # Output debug information
        print("\n=== Final Point Check ===")
        print(f"Final point: {final_point}")
        if within_bounds:
            print("The final point is within the mesh bounds.")
        else:
            print("The final point is OUTSIDE the mesh bounds.")
        
        print(f"Total distance from origin to destination: {total_direct_distance:.4f}")
        print(f"Sum of distances in voxels: {total_voxel_distance:.4f}")
        print(f"Distance from final point to destination: {final_to_destination_distance:.4f}")
        print(f"Total path length: {total_voxel_distance + final_to_destination_distance:.4f}")
        print("==========================\n")
        
        # Check if the total distance matches the expected distance
        if np.isclose(total_direct_distance, total_voxel_distance + final_to_destination_distance):
            print("Success: The total distance matches the sum of the voxel distances and final-to-destination distance!")
        else:
            print("Warning: The total distance does NOT match!")
        print("==========================\n")

def generate_voxel_id(mesh):
    """
    Generates an array of shape = mesh.shape containing the voxel_ids, which are assigned following C style 
    Last axis changing faster (Z, i.e. k indices).
    """
    num_voxels = np.prod(mesh.shape)
    voxel_id = np.arange(num_voxels).reshape(mesh.shape)
    return voxel_id

def calculate_path(args):
    """
    arranger function that is submited to the executor for parallel computing
    it takes the arguments to be handled to the tracker in shape of a tuple.
    Type and sizes of the tuple elements are not checked.
    """
    iteration_results = [] 
    detector_rel_coord, voxel_id,voxel_center= args 
    voxel_distances = _tracker.track_particle(origin = voxel_center, destination = detector_rel_coord)
    indices_list = [indices for indices, _ in voxel_distances]
    distances_list = [distance for _, distance in voxel_distances]
    indices_array = np.array(indices_list)
    distances_array = np.array(distances_list)
    if not voxel_distances:
        """
        This is super technical debugging stuff
        """
        print ('ayyyyyyyyyyyyyyyyyyyy')

        iteration_results = [{
            'i': 666,
            'j': 666,
            'k': 666,
            'd': 666,
            'voxel_id': voxel_id
        }]
    
    else:
        iteration_results = [{
            'i': i,
            'j': j,
            'k': k,
            'd': d,
            'voxel_id': voxel_id
        } for i, j, k, d in zip(indices_array[:, 0], indices_array[:, 1], indices_array[:, 2], distances_array)]
    return iteration_results



#Init of the flow here
def parallel_path_calculation(mesh,detector_rel_coords,mesh_folder_path,unique_ids,chunk_size=25000,max_workers=10,start_method="spawn"):
    """
    Chunk_size optimum is about 15 000 for POSIX in spawn mode
    In windows, Spawn by default, there is no need to do chunks if workers <= 4
    """
    # Use a local context rather than changing the multiprocessing start method
    # globally. 'fork' may be unsafe with JAX/TF or already-running threads.
    if start_method not in mp.get_all_start_methods():
        raise ValueError(f"Unsupported multiprocessing start method: {start_method}")
    context = mp.get_context(start_method)

    detector_position_counter = 0
    tracker = ParticleTracker(
        voxel_size=mesh.attrs['spacing'], 
        mesh_min=[mesh.coords['x'][0]-mesh.attrs['spacing'][0]/2 , mesh.coords['y'][0]-mesh.attrs['spacing'][1]/2  , mesh.coords['z'][0] -mesh.attrs['spacing'][2]/2 ], 
        mesh_max=[mesh.coords['x'][-1]+mesh.attrs['spacing'][0]/2  , mesh.coords['y'][-1] +mesh.attrs['spacing'][1]/2 , mesh.coords['z'][-1] +mesh.attrs['spacing'][2]/2 ]
        )


    voxel_ids = generate_voxel_id(mesh)
    # Reuse the same workers for all detector positions and voxel chunks.
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=max_workers,
        mp_context=context,
        initializer=init_tracker,
        initargs=(tracker,),
    ) as executor:
        for detector_rel_coord in detector_rel_coords:
            data_list = []
            start_time = time.time()
            detector_csv_path = os.path.join(mesh_folder_path, unique_ids[detector_position_counter])
            detector_position_counter += 1
            print(f'CALCULATING PATHS FOR DETECTOR POSITION:  {detector_position_counter} out of {len(detector_rel_coords)}')
            voxel_args = [
                (
                    detector_rel_coord,
                    voxel_ids[i, j, k],
                    [mesh.coords[dim][index].values for dim, index in zip(['x', 'y', 'z'], [i, j, k])],
                )
                for i in range(mesh.shape[0])
                for j in range(mesh.shape[1])
                for k in range(mesh.shape[2])
            ]
            for start in range(0, len(voxel_args), chunk_size):
                chunk = voxel_args[start:start + chunk_size]
                # map() preserves input order and batches tasks for IPC efficiency.
                results = executor.map(calculate_path, chunk, chunksize=32)
                for result in tqdm(results, total=len(chunk)):
                    data_list.extend(result)

            print('SAVING VOXELS FOR DETECTOR POSITION')
            Detector_POV = pl.DataFrame(data_list)
            Detector_POV.write_csv(detector_csv_path + '.csv')
            print('Elapsed time for position of Detector:', time.time() - start_time, 'seconds')
            gc.collect()

def debug():
    voxel_size = [19.2, 19.2, 19.2]  # Define voxel size (dx, dy, dz)
    mesh_min = [-288-19.2/2, -451.2-19.2/2, -288.0-19.2/2]  # Mesh lower bounds (x_min, y_min, z_min)
    mesh_max = [288+19.2/2, 451.2+19.2/2, 288.0+19.2/2]  # Mesh upper bounds (x_max, y_max, z_max)
    
    tracker = ParticleTracker(voxel_size, mesh_min, mesh_max)
    origin = [1e-12,412.51,1e-5]  # Particle origin at the center of voxel (5, 5, 5)
    destination = [.0,412.5,-437 ]  # Particle 
        
    
    voxel_distances = tracker.track_particle(origin, destination,debug=True)
    
    for voxel, distance in voxel_distances:
        print(f"Voxel: {voxel}, Distance: {distance:.4f}")
    
    print(f'Total voxels crossed: {len(voxel_distances)}')
    
    
    

   

