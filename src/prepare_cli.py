from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parent

import os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
 
from src.config import load_config
import time 
 
import sys
"""
Spatial Activity Reconstruction Using Machine-learning Algorithms for Nuclear waste
"""

class Saruman():
    #default_context_dict={}   
    def __init__(self,context_dict=None, config_path=None):
            # Load and validate parameters; scientific modules are wired in later stages.
            self.config = load_config(config_path)
    
    def create_batch(self,spectrum_folder,ct_filepath):
        from src.radwaste_engine.radwaste_engine import WasteBatch
        from src import forward_model
        from src.tools import io_tools
        start_time= time.time()
        print ("Initializing waste package\n")
        waste_batch =WasteBatch(ct_filepath, config=self.config) #Add context variables via  arguments or set_context method
        print ("\nReading spectrum files...")
        waste_batch.spectrum_analyzed,waste_batch.spc_time,waste_batch.energies_detected, waste_batch.isotopes_detected,_= io_tools.read_spectrum_files(spectrum_folder,False)
   
        forward_model.get_det_efficiency(waste_batch)
 
        forward_model.adu2lac(waste_batch) #add adu2lac bypass as method
        
        #Debug   
        end_time = time.time()
        exec_time = -start_time +end_time
        print(f"Execution time: {exec_time:.4f} seconds")
        return waste_batch
    
   
    @staticmethod
    def load(filename):
        """
        Loads a WasteBatch instance from a file.
        :param filename: Path to the file to load.
        :return: Loaded WasteBatch instance.
        """
        from src.radwaste_engine.radwaste_engine import WasteBatch
        import h5py
        import xarray as xr
        batch = WasteBatch(ct_filename=None)  # si quieres evitar recrear mesh original
        with h5py.File(filename, "r") as f:
            # ----- mesh -----
            if "mesh_values" in f:
                mesh_vals = f["mesh_values"][()]
                coords = {key.replace("mesh_coords_", ""): f[key][()]
                          for key in f if key.startswith("mesh_coords_")}
                batch.mesh = xr.DataArray(mesh_vals, coords=coords)
 
            # ----- big arrays -----
            if "spc_time" in f:
                batch.spc_time =f["spc_time"][()]
            if "A" in f:
                batch.A = f["A"][()]
            if "spectrum_analyzed" in f:
                batch.spectrum_analyzed = f["spectrum_analyzed"][()]
            if "detector_efficiencies" in f:
 
                batch.detector_efficiencies = f["detector_efficiencies"][()]
            if "transmission" in f:
                batch.transmission = f["transmission"][()]
            else:
                batch.transmission = f.attrs.get("transmission", False)
 
            # ----- lists -----
            if "energies_detected" in f:
                batch.energies_detected = f["energies_detected"][()].tolist()
            if "isotopes_detected" in f:
                batch.isotopes_detected = [s.decode("utf-8") for s in f["isotopes_detected"][()]]
        print('Waste packages succesfully loaded')
        batch.describe()
        return batch
       
    
 
        
    def save(self,waste_batch, filename):
                import h5py
                import numpy as np
                with h5py.File(f'{filename}.h5', "w") as f:
                    # ----- mesh -----
                    if waste_batch.mesh is not None:
                        f.create_dataset("mesh_values", data=waste_batch.mesh.values,compression='gzip',compression_opts=4)
                        # saves dimensions coords
                        for dim, coord in waste_batch.mesh.coords.items():
                            f.create_dataset(f"mesh_coords_{dim}", data=coord.values,compression='gzip',compression_opts=4)
                
                    # ----- arrays -----
                    if waste_batch.A is not None:
                        f.create_dataset("A", data=waste_batch.A)
                    if waste_batch.spectrum_analyzed is not None:
                        f.create_dataset("spectrum_analyzed", data=waste_batch.spectrum_analyzed,compression='gzip',compression_opts=4)
                    if waste_batch.detector_efficiencies is not None:
 
                        f.create_dataset("detector_efficiencies", data=waste_batch.detector_efficiencies,compression='gzip',compression_opts=4)
                    if isinstance(waste_batch.transmission, np.ndarray):
                        f.create_dataset("transmission", data=waste_batch.transmission,compression='gzip',compression_opts=4)
                    else:
                        f.attrs["transmission"] = waste_batch.transmission
                    if waste_batch.spc_time is not None:
                        f.create_dataset("spc_time", data=waste_batch.spc_time,compression='gzip',compression_opts=4)
                        
                    # ----- lists -----
                    dt = h5py.string_dtype(encoding='utf-8')
                    if waste_batch.energies_detected:
                        f.create_dataset("energies_detected", data=np.array(waste_batch.energies_detected, dtype=float),compression='gzip',compression_opts=4)
                    if waste_batch.isotopes_detected:
                        f.create_dataset("isotopes_detected", data=np.array(waste_batch.isotopes_detected, dtype=dt),compression='gzip',compression_opts=4)


      
    def analyze_batch(self,waste_batch):
        #Check if there's activity already saved in the object
        #call solvers depending on context by WasteBatch. Redirect flow by flavors 
        #pkl waste_batch
        if True: #if waste_batch.A:
            while True:
                choice = input("This waste batch has already been analyzed.\nDo you wish to venture forth? (Yes/No): ").strip().lower()
    
                if choice == "yes":
                    print("Very well... we shall proceed, for the task must not falter.")
                    #solver.solve(waste_batch)
                    break  # or continue with the rest of your program
                elif choice == "no":
                    print("So, you choose to end this endeavor. Against the will of power, resistance is but folly. The program ceases at your command.")
                    sys.exit()
                else:
                    print("The answer is unclear... only 'Yes' or 'No' shall suffice. Choose wisely.")

        else:
            #solver.solve(waste_batch)
            pass
        
    def show_results(self,waste_batch):
        #results of the solver 
        pass         
     
       



def _prepare_cli(args):
    """Build one HDF5 package, using the existing scientific preparation pipeline."""
    from src.saruman_messages import announce

    # Resolve user inputs relative to the caller's working directory, not SARUMAN.
    ct = Path(args.ct).expanduser().resolve()
    spectra = Path(args.spectra).expanduser().resolve()
    output = Path(args.output).expanduser().resolve()
    if output.suffix.lower() != ".h5":
        output = output.with_name(output.name + ".h5")

    wizard = Saruman(config_path=args.config)
    if args.detector:
        detector = Path(args.detector).expanduser().resolve()
        wizard.config["paths"]["detector_coordinates_file"] = detector
    else:
        detector = Path(wizard.config["paths"]["detector_coordinates_file"])

    for label, path, kind in (("CT", ct, "file"),
                              ("ASGS spectra", spectra, "dir"),
                              ("Detector coordinates", detector, "file")):
        valid = path.is_file() if kind == "file" else path.is_dir()
        if not valid:
            raise FileNotFoundError(f"{label} {kind} not found: {path}")
    if output.exists() and not args.overwrite:
        raise FileExistsError(f"Output already exists: {output} (use --overwrite to replace it)")
    if not output.parent.is_dir() and not args.dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)

    cfg = wizard.config
    ct_cfg = cfg["ct_processing"]
    path_cfg = cfg["path_calculation"]
    print('SARUMAN: "The seeing stone is ready. Let the hidden paths be revealed."', flush=True)
    print(f"CT:                 {ct}")
    print(f"ASGS spectra:       {spectra}")
    print(f"Detector positions: {detector}")
    print(f"Output package:     {output}")
    print(f"CT downsampling:    {ct_cfg['downsampling_method']}")
    print(f"Grid size:          {ct_cfg['grid_size']}")
    print(f"Grid spacing (mm):  {ct_cfg['grid_spacing_mm']}")
    print(f"CT shift (mm):      {ct_cfg['shift_mm']}")
    print(f"Path workers:       {path_cfg['max_workers']} ({path_cfg['start_method']})")
    print(f"Scenario database:  {cfg['paths']['data_dir']}")
    print("Detector surrogate: neural network")
    print(f"NN model:           {cfg['detector_surrogate']['nn']['model_path']}")
    print(f"CT2LAC RF model:    {cfg['ct2lac']['rf_model_path']}", flush=True)

    if args.dry_run:
        print('SARUMAN: "I have surveyed the realm. No calculations shall begin."')
        print("DRY RUN: paths checked; CT, ray tracing, surrogate inference and HDF5 writing skipped.")
        return

    batch = wizard.create_batch(str(spectra), str(ct))
    # The legacy save() method appends .h5 itself. Pass the stem to avoid .h5.h5.
    wizard.save(batch, str(output.with_suffix("")))
    announce("prepare_done")
    print(f"HDF5 written: {output}")


def cli_main():
    import argparse
    parser = argparse.ArgumentParser(
        prog="saruman",
        description="SARUMAN — Spatial Activity Reconstruction Using Machine-learning Algorithms for Nuclear Waste",
    )
    commands = parser.add_subparsers(dest="command", required=True)
    prepare = commands.add_parser("prepare", help="Generate a drum HDF5 from CT and ASGS spectra")
    prepare.add_argument("--ct", required=True, help="Input CT reconstruction (.mhd)")
    prepare.add_argument("--spectra", required=True, help="Directory of ASGS spectra")
    prepare.add_argument("--detector", help="Detector coordinates CSV (defaults to config.yaml)")
    prepare.add_argument("--output", required=True, help="Output HDF5 filename")
    prepare.add_argument("--config", help="User YAML (defaults to ./config.yaml or installed defaults)")
    prepare.add_argument("--dry-run", action="store_true", help="Check inputs and display settings without calculating")
    prepare.add_argument("--overwrite", action="store_true", help="Allow replacing an existing output HDF5")
    args = parser.parse_args()
    try:
        if args.command == "prepare":
            _prepare_cli(args)
    except (ValueError, FileNotFoundError, FileExistsError, PermissionError) as exc:
        parser.exit(2, f'SARUMAN: "Even my wisdom requires sound foundations."\nERROR: {exc}\n')


if __name__ == "__main__":
    cli_main()
