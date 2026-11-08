#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  7 13:52:10 2025

@author: vcmolina@sck.be
"""
import numpy as np
class ParticleTracker:
    def __init__(self, voxel_size, mesh_min, mesh_max):
        self.voxel_size = np.array([19.2, 19.2, 19.2])  # Define voxel size (dx, dy, dz)
        self.mesh_min = np.array([-288-19.2/2, -451.2-19.2/2, -288.0-19.2/2] ) # Mesh lower bounds (x_min, y_min, z_min)
        self.mesh_max = np.array([288+19.2/2, 451.2+19.2/2, 288.0+19.2/2]  )# Mesh upper bounds (x_max, y_max, z_max)
        

    def track_particle(self, origin, destination, debug=False):
     # Initialize particle tracking information
     origin = np.array(origin)
     destination = np.array(destination)
     
     # Calculate the direction as the normalized vector from origin to destination
     direction = destination - origin
     direction /= np.linalg.norm(direction)  # Normalize the direction vector
     
     # Find the initial voxel the particle is in
     current_voxel = np.floor((origin - self.mesh_min) / self.voxel_size).astype(int)
   
     # Determine step direction (+1 or -1) based on particle direction
     step = np.sign(direction).astype(int)
     
     # Calculate t_max (distance to next voxel boundary in each axis)
     t_max = ((self.voxel_size * (current_voxel + (step > 0)) + self.mesh_min - origin) / direction)
     
     # Calculate t_delta (distance across one voxel in each direction)
     t_delta = np.abs(self.voxel_size / direction)
     
     # Track distances and voxels traversed
     total_distance = 0
     voxel_distances = []
     
     # Distance to first voxel boundary
     distance_in_first_voxel = np.min(t_max)  # Shortest distance to leave the first voxel
     
     # Include the initial voxel and the distance traversed inside it
     #voxel_distances.append((tuple(current_voxel), distance_in_first_voxel))
     total_distance += distance_in_first_voxel
     
     # Update t_max for the next voxel boundary crossing
     next_axis = np.argmin(t_max)
     t_max[next_axis] += t_delta[next_axis]
     current_voxel[next_axis] += step[next_axis]
     
     # Now continue tracking for the remaining voxels
     while np.all(current_voxel >= 0) and np.all(abs((total_distance + origin) * direction) < abs(self.mesh_max)):
         
         # Find the next axis to cross (smallest t_max)
         next_axis = np.argmin(t_max)
         
         # Distance to cross to the next voxel
         distance = t_max[next_axis] - total_distance
         total_distance += distance
         
         # Check if the next voxel is within the mesh dimensions
         next_voxel = current_voxel.copy()
         next_voxel[next_axis] += step[next_axis]
         
         
         # Record the voxel and distance traversed
         voxel_distances.append((tuple(current_voxel), distance))
         
         # Move to the next voxel
         current_voxel = next_voxel
         
         # Update t_max for the next voxel boundary crossing
         t_max[next_axis] += t_delta[next_axis]
     
     #if debug:
        # self.check_final_point(final_point, destination, voxel_distances, origin)
     
     return voxel_distances
origin = [5.68e-14,-412.51,5.68e-14]  # Particle origin at the center of voxel (5, 5, 5)
destination = [437.,  -412.5 ,   0.  ]  # Particle 
voxel_size = [19.2, 19.2, 19.2]  # Define voxel size (dx, dy, dz)
mesh_min = [-288-19.2/2, -451.2-19.2/2, -288.0-19.2/2]  # Mesh lower bounds (x_min, y_min, z_min)
mesh_max = [288+19.2/2, 451.2+19.2/2, 288.0+19.2/2]  # Mesh upper bounds (x_max, y_max, z_max)
tracker=ParticleTracker(voxel_size,mesh_min,mesh_max)
voxel_distances = tracker.track_particle(origin, destination)
for voxel, distance in voxel_distances:
      print(f"Voxel: {voxel}, Distance: {distance:.4f}")
  