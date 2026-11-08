 
import uuid
import numpy as np
import os
import pandas as pd 

def generate_uuid (parameter1,parameter2,parameter3,parameter4):
    identifier_string = f"{parameter1}_{parameter2}_{parameter3}_{parameter4}"
    unique_id = uuid.uuid5(uuid.NAMESPACE_OID, identifier_string)
    return str(unique_id)

def log_new_mesh (unique_id):
    data_directory = './src/data'   
    # Create the full path for the folder
    folder_path = os.path.join(data_directory, str(unique_id))
    print(folder_path)
    # Check if the folder already exists
    if not os.path.exists(folder_path):
        # If the folder does not exist, create it
        os.makedirs(folder_path)
        print(f"Folder for ID {unique_id} created successfully.")
    else:
        print(f"Folder for ID {unique_id} already exists.")
    

 




def read_spectrum_files(SPECTRA_DIR, Synthetic=True):
    ### the reader should check if the detector positions and the spectrum files match,
    ### and then load per detector position the data of that spectrum (raw)
    ### then, the "spectrum_analyzer" should "analyze" deadtimes, FEPE, UNCERT, warnings, etc 
    
    # --- Tags of interest ---
    num_peak_tag = 'Count'
    counts_tag = 'Sq'
    sigma_tag = 'Sq_sig'
    E_tag = 'Ident_En_E'
    spc_time_tag = 'spc_time'
    nuclide_tag = 'Ident_name_nuclide'

    # --- Parameters to modify by the context editor ---
    NUM_SEGMENTS = 15
    NUM_SECTORS = 12

    def parse_peak_file(filename):
        with open(filename, 'r') as f:
            lines = f.readlines()

        data = {}
        current_section = None
        for line in lines:
            line = line.strip()
            if not line or line.startswith(";"):
                continue
            if line.startswith("[") and line.endswith("]"):
                current_section = line.strip("[]")
                data[current_section] = {}
            elif current_section and "=" in line:
                key, val = line.split("=", 1)
                data[current_section][key.strip()] = val.strip()

        n_peaks = int(data["Main"][num_peak_tag])
        spc_time = float(data.get("Peak_1", {}).get(spc_time_tag, 0))

        peaks = []
        for i in range(1, n_peaks + 1):
            peak = data.get(f"Peak_{i}", {})
            peaks.append({
                "Energy": float(peak.get(E_tag, np.nan)),
                "Sq": float(peak.get(counts_tag, 0)),
                "Sigma": float(peak.get(sigma_tag, 0)),
                "Nuclide": peak.get(nuclide_tag, None)
            })

        return peaks, spc_time

    def parse_detector(fname):
        core = os.path.splitext(fname)[0]
        if len(core) < 4:
            raise ValueError(f"Foolish mortal! '{fname}' dares to defy Saruman’s code!")
        last4 = core[-4:]
        segment = int(last4[:2])
        sector = int(last4[2:])
        return segment, sector

    def parse_folder_aligned(folder, num_segments=NUM_SEGMENTS, num_sectors=NUM_SECTORS, skip_sector0=True):
        all_peaks = []
        all_spc_time = []
        file_info = []

        # --- Parse all files ---
        for fname in os.listdir(folder):
            if not fname.lower().endswith(".asr"):
                continue

            segment, sector = parse_detector(fname)
            if skip_sector0 and sector == 0:
                continue
            if segment > num_segments or sector > num_sectors:
                continue

            peaks, spc_time = parse_peak_file(os.path.join(folder, fname))
            all_peaks.append(peaks)
            all_spc_time.append(spc_time)
            file_info.append({"filename": fname, "segment": segment, "sector": sector})

        # --- Master list of energies and nuclides ---
        master_dict = {}
        for peaks in all_peaks:
            for p in peaks:
                E = p["Energy"]
                if E not in master_dict:
                    master_dict[E] = p["Nuclide"]
        master_energies = sorted(master_dict.keys())
        master_nuclides = [master_dict[E] for E in master_energies]

        # --- Mean spc_time for undetected count files ---
        mean_spc_time = np.mean(all_spc_time) if all_spc_time else 0

        # --- Build aligned DataFrame with all segment-sector combinations ---
        rows = []
        for seg in range(1, num_segments + 1):
            for sec in range(1, num_sectors + 1):
                idx = next(
                    (i for i, info in enumerate(file_info)
                     if info["segment"] == seg and info["sector"] == sec),
                    None
                )
                row = {"segment": seg, "sector": sec}

                if idx is not None:
                    peaks = all_peaks[idx]
                    spc_time = all_spc_time[idx]
                    peak_dict = {p["Energy"]: p for p in peaks}
                    row["filename"] = file_info[idx]["filename"]
                else:
                    peaks = []
                    spc_time = mean_spc_time  # <-- usar media
                    peak_dict = {}
                    row["filename"] = None

                row["spc_time"] = spc_time

                for i, E in enumerate(master_energies, start=1):
                    row[f"Energy_{i}"] = E
                    row[f"Sq_{i}"] = peak_dict[E]["Sq"] if E in peak_dict else 0
                    row[f"Sigma_{i}"] = peak_dict[E]["Sigma"] if E in peak_dict else 0
                    row[f"Nuclide_{i}"] = master_dict[E]

                rows.append(row)

        df = pd.DataFrame(rows)
        df = df.sort_values(by=["segment", "sector"]).reset_index(drop=True)
        return df, master_energies, master_nuclides

    # ---- USAGE ----
    folder = SPECTRA_DIR
    aligned_df, energies, nuclides = parse_folder_aligned(folder)

    num_energies = len(energies)
    num_detectors = len(aligned_df)
    counts = np.zeros((num_energies, num_detectors, 2))

    # --- Counts matrix: energy_index x detector_index ---
    for i in range(num_energies):
        counts[i, :, 0] = aligned_df[f"Sq_{i+1}"].to_numpy()  # counts
        counts[i, :, 1] = aligned_df[f"Sigma_{i+1}"].to_numpy()  # sigmas

    # --- spc_time vector by detector_index ---
    spc_times = aligned_df["spc_time"].to_numpy()

    return counts, spc_times, energies, nuclides,aligned_df
