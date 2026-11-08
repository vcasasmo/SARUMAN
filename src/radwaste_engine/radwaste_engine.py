from src.radwaste_engine.xr_processor import process_ct
import numpy as np
from src.tools.io_tools import  generate_uuid
import os
import polars as pl
import sys
import src.radwaste_engine.path_calculator
 
#from path_calculator import path_calculator_routine
from tqdm import tqdm
# Detector coordinates and data root are supplied by the loaded configuration.
# No global, process-wide mutable paths are needed.

def mesh_init(image_xarray):
    # Extract origin and voxel size from attributes
    origin = np.array(image_xarray.attrs['origin'])
    voxel_size = np.array(image_xarray.attrs['spacing'])
    shape = image_xarray.shape
    # Create a grid of coordinates using numpy
    x_coords = np.linspace(origin[0], origin[0] + voxel_size[0] * shape[0], shape[0], endpoint=False)
    y_coords = np.linspace(origin[1], origin[1] + voxel_size[1] * shape[1], shape[1], endpoint=False)
    z_coords = np.linspace(origin[2], origin[2] + voxel_size[2] * shape[2], shape[2], endpoint=False)
    # Update coordinates in the DataArray
    updated_xarray = image_xarray.assign_coords(
        x=('x', x_coords.flatten()),
        y=('y', y_coords.flatten()),
        z=('z', z_coords.flatten())
    )
 
    
    updated_xarray.attrs['mesh_id'] = generate_uuid(image_xarray.attrs['origin'],image_xarray.attrs['spacing'],image_xarray.shape,'mesh')

   
    return updated_xarray  



def check_database_for_mesh(mesh, data_dir):
    mesh_id = mesh.attrs.get('mesh_id', None)

    if mesh_id:
        MESH_DIR=os.path.expanduser(os.fspath(data_dir))
        mesh_folder_path = os.path.join(MESH_DIR, mesh_id)
        # Check if the folder with the mesh_id exists
        if os.path.exists(mesh_folder_path) and os.path.isdir(mesh_folder_path):
            print('A mesh entry with this characteristics was found')
            return mesh_folder_path
        if not os.path.exists(mesh_folder_path):
            print ('This mesh was not found in datafiles. Creating Entry ...')
            os.makedirs(mesh_folder_path)
            print(f"Entry for ID {mesh_id} created successfully.")
            return mesh_folder_path
    else:
        print('Mesh was not properly initialized: ID missing.')
        # Return False if the mesh_id attribute is not present in the mesh
        sys.exit("Program terminated due to missing mesh ID.")

def init_detector_positions(mesh, mesh_folder_path, detector_coordinates_file, path_settings):

    detector_coordinates_raw = pl.read_csv(os.fspath(detector_coordinates_file))

    def rotate_detector_coords(detector_coordinates_raw, theta=180):
        """
        Rotate detector coordinates around the Y-axis by theta degrees.
        
        Parameters:
            detector_coordinates_raw (pl.DataFrame): DataFrame with columns 'X', 'Y', 'Z'.
            theta (float): Angle in degrees to rotate around the Y-axis. Default is 180 degrees.
            
        Returns:
            pl.DataFrame: Rotated detector coordinates DataFrame.
        """
        # Convert theta from degrees to radians
        theta_rad = np.deg2rad(theta)
        print(f"Rotating drum  {theta} degrees ...  ")
        
        # Extract X, Y, Z coordinates from the DataFrame
        points = detector_coordinates_raw[['x', 'y', 'z']].to_numpy()
        #print(detector_coordinates_raw)
        # Apply rotation
        cos_theta = np.cos(theta_rad)
        sin_theta = np.sin(theta_rad)
        rotation_matrix = np.array([[cos_theta, 0, sin_theta],
                                    [0, 1, 0],
                                    [-sin_theta, 0, cos_theta]])
        rotated_points = np.round(np.dot(points, rotation_matrix.T),2)
        
        # Update the DataFrame with rotated coordinates
        rotated_coordinates = pl.DataFrame({
    'x': rotated_points[:,0],
    'y': rotated_points[:,1],
    'z': rotated_points[:,2]
})
     
        return rotated_coordinates
    detector_coordinates=detector_coordinates_raw 
    #detector_coordinates = rotate_detector_coords(detector_coordinates_raw)
    #The order inside the csv for the positions has to be the same as for the spectroscopy files
    missing_positions=[]
    missing_positions_ids=[]

 
    for position in tqdm(range(len(detector_coordinates)), desc='Checking detector positions within the mesh'):
        det_x,det_y,det_z = detector_coordinates.row(position)
        unique_id = generate_uuid(det_x,det_y,det_z,'detector')
        # Check if the CSV file with the unique_id exists in the mesh_folder_path
        detector_csv_path = os.path.join(mesh_folder_path, f'{unique_id}.csv')
        
        if not os.path.exists(detector_csv_path):
    
  
            # Create a CSV file for the detector in the mesh folder
            missing_positions.append(np.array([det_x, det_y, det_z]))
            missing_positions_ids.append(unique_id)

    if missing_positions:
        print('The following detector positions were not computed for this mesh:')  
        for position in missing_positions:
            print (f' {position} ')
        src.radwaste_engine.path_calculator.parallel_path_calculation(mesh=mesh,detector_rel_coords=missing_positions,
                                                                       unique_ids=missing_positions_ids, mesh_folder_path=mesh_folder_path,
                                                                       chunk_size=int(path_settings['chunk_size']),
                                                                       max_workers=int(path_settings['max_workers']),
                                                                       start_method=path_settings['start_method']) 
   
 
    return detector_coordinates


###############DRUM INIT###################################  
class WasteBatch:
 
        #TODO: maybe add a method to let it add more detector positions once it's instantiated 
    
 
    def __init__(self, ct_filename=None, config=None):
        if config is None:
            from src.config import load_config
            config = load_config()
        self.config = config
        if ct_filename is not None:
            
            ct_scan = process_ct(ct_filename, config=config)
            self.mesh = mesh_init(image_xarray=ct_scan)
            self.mesh_folder_path = check_database_for_mesh(self.mesh, config['paths']['data_dir'])
            self.detector_positions = init_detector_positions(
                mesh=self.mesh, mesh_folder_path=self.mesh_folder_path,
                detector_coordinates_file=config['paths']['detector_coordinates_file'],
                path_settings=config['path_calculation'],
            )
        else:
    
            self.mesh = None
            self.mesh_folder_path = None
            self.paths = None
            self.detector_positions = None

        # Always required attributes
        self.spectrum_analyzed = None
        self.energies_detected = []
        self.detector_efficiencies = None
        self.isotopes_detected = []
        self.transmission = False
        self.spc_time = None
        self.A = None  
        self.metadata = {}
        
 
    def describe(self):
        """
        Shows description of drum
        """
        print("=== WasteBatch Summary ===")
        
        # Mesh
        if self.mesh is not None:
            print(f"mesh: DataArray, shape={self.mesh.shape}, dims={self.mesh.dims}")
        else:
            print("mesh: None")
        
        # Spectrum analyzed
        if self.spectrum_analyzed is not None:
            print(f"spectrum_analyzed: ndarray, shape={self.spectrum_analyzed.shape}")
        else:
            print("spectrum_analyzed: None")
        
        # Energies detected
        print(f"energies_detected (keV): {self.energies_detected}")
        
        # Isotopes detected
        print(f"isotopes_detected: {self.isotopes_detected}")
        
        # Detector efficiencies
        if self.detector_efficiencies is not None:
            print(f"detector_efficiencies: ndarray, shape={self.detector_efficiencies.shape}")
        else:
            print("detector_efficiencies: None")
        
        # Transmission
        if isinstance(self.transmission, np.ndarray):
            print(f"transmission: ndarray, shape={self.transmission.shape}")
        else:
            print(f"transmission: {self.transmission}")
        
        # A
        if self.A is not None:
            print(f"A: ndarray, shape={self.A.shape}")
        else:
            print("A: None")
        print(f'Average measurement time: {np.round(np.mean(self.spc_time), 2)} s')
        # Metadata
        print(f"metadata: dict, {len(self.metadata)} entries")
        
        print("==========================")    

    
if __name__ == "__main__":
    drum = WasteBatch()


#### TODO: work on the docs before moving out
 
#####TODO: work on the simmetry detection in path_calculator

    


