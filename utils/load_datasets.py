import tarfile
import os
import pandas as pd
from loguru import logger
from collections import Counter
from tqdm import tqdm
from fairchem.core.datasets import AseDBDataset
import h5py
import glob
import socket

class OMol25Loader:
    def __init__(self, base_path="data/OMol25/", split="neutral_val"):

        hostname = socket.gethostname() # Will be applicable once we have data on Niflheim
        is_niflheim = "fysik.dtu.dk" in hostname or any(x in hostname for x in ["surt", "sylg", "slid", "svol"])
        if is_niflheim:
            self.base_path = base_path
        else:
            self.base_path = base_path

        self.split = split
        self.tar_path = os.path.join(self.base_path, f"{self.split}.tar")
        self.output_path = os.path.join(self.base_path, self.split)
        self.dataset = None

    def _prepare_data(self):
        """Checks if data is extracted; if not, extracts the tar file."""
        if not os.path.exists(self.output_path) or len(os.listdir(self.output_path)) == 0:
            logger.info(f"Tar file not found extracted. Starting extraction: {self.tar_path}")
            try:
                os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
                with tarfile.open(self.tar_path, "r") as tar:
                    tar.extractall(path=os.path.join(self.base_path, "output"))
                logger.info("Extraction completed successfully.")
            except Exception as e:
                logger.error(f"Failed to extract tar file: {e}")
                raise
        else:
            logger.info(f"Found existing extracted data at {self.output_path}")
        
        # Initialize the Fairchem dataset
        try:
            self.dataset = AseDBDataset({"src": self.output_path})
            logger.info(f"Fairchem dataset loaded with {len(self.dataset)} entries.")
        except Exception as e:
            logger.error(f"Error initializing AseDBDataset: {e}")
            raise

    def get_omol25(self, sample_size=None):
        """
        Processes the dataset and returns a Pandas DataFrame and a Symbol Counter.
        :param sample_size: Integer. If provided, only processes the first N entries.
        """
        if self.dataset is None:
            self._prepare_data()

        total_to_process = sample_size if sample_size else len(self.dataset)
        logger.info(f"Beginning data processing for {total_to_process} entries.")
        
        data_list = []
        symbol_counts = Counter()

        # Iterate through the dataset
        for i in tqdm(range(total_to_process), desc="Extracting OMol25 Properties"):
            try:
                atoms = self.dataset.get_atoms(i)
                
                # Update symbol counts
                symbol_counts.update(atoms.get_chemical_symbols())
                
                # Basic info
                entry = {
                    "index": i,
                    "num_atoms": len(atoms),
                    "formula": atoms.get_chemical_formula(),
                    "potential_energy": atoms.get_potential_energy() if atoms.get_potential_energy() else None
                }
                
                info_keys = [
                    'num_electrons', 'n_scf_steps', 'n_basis', 
                    'homo_lumo_gap', 'nl_energy'
                ]
                
                for key in info_keys:
                    val = atoms.info.get(key)
                    if key == 'homo_lumo_gap' and isinstance(val, (list, tuple, pd.Series)) and len(val) > 0:
                        entry[key] = val[0]
                    else:
                        entry[key] = val
                
                data_list.append(entry)
            except Exception as e:
                # We log as warning to continue processing other files if one fails
                logger.warning(f"Error processing entry at index {i}: {e}")

        logger.info("Data extraction finished. Creating DataFrame.")
        df = pd.DataFrame(data_list)
        
        # Log a summary of what was found
        logger.info(f"DataFrame created with {len(df)} rows.")
        logger.info(f"Detected elements: {list(symbol_counts.keys())}")
        
        return df, symbol_counts


class MaterialsProjectLoader:
    def __init__(self, base_path="data/Materials Project/", file_name="data.hdf5"):

        hostname = socket.gethostname() # Will be applicable once we have data on Niflheim
        is_niflheim = "fysik.dtu.dk" in hostname or any(x in hostname for x in ["surt", "sylg", "slid", "svol"])
        if is_niflheim:
            self.base_path = base_path
        else:
            self.base_path = base_path

        self.file_path = os.path.join(self.base_path, file_name)
        self.df = pd.DataFrame()

    def load_data(self):

        if "data.hdf5" not in os.listdir(self.base_path):
            with tarfile.open(self.base_path+"materials-project.tar", "r") as tar:
                tar.extractall(path=self.base_path)

        if not os.path.exists(self.file_path):
            logger.error(f"File not found: {self.file_path}")
            raise FileNotFoundError(f"Could not find HDF5 file at {self.file_path}")

        try:
            with h5py.File(self.file_path, "r") as f:
                keys = list(f.keys())
                if not keys:
                    return pd.DataFrame()

                main_group_key = keys[0]
                group_obj = f[main_group_key]

                if not isinstance(group_obj, h5py.Group):
                    return pd.DataFrame({main_group_key: group_obj[()]})

                # 1. First pass: identify the most common array length
                lengths = []
                for member_name in group_obj.keys():
                    ds = group_obj[member_name]
                    if ds.ndim == 1:
                        lengths.append(len(ds))
                
                if not lengths:
                    logger.warning("No 1D datasets found.")
                    return pd.DataFrame()

                # Find the most frequent length (the number of materials)
                primary_length = max(set(lengths), key=lengths.count)
                logger.info(f"Detected primary dataset length: {primary_length}")

                # 2. Second pass: only load datasets matching that length
                data_dict = {}
                for member_name in group_obj.keys():
                    ds = group_obj[member_name]
                    if ds.ndim == 1 and len(ds) == primary_length:
                        data_dict[member_name] = ds[()]
                    else:
                        logger.debug(f"Skipping {member_name}: Length {len(ds)} != {primary_length}")

                self.df = pd.DataFrame(data_dict)
                logger.info(f"Successfully loaded {len(self.df.columns)} columns.")
                
            return self.df

        except Exception as e:
            logger.error(f"Error: {e}")
            raise

    def get_summary(self):
        if self.df.empty:
            print("No data loaded.")
            return
        print(f"\n--- Loaded {len(self.df)} Materials ---")
        print(self.df.info())

class OMat24Loader:
    def __init__(self, base_path="data/OMat24/", split="train"):
        """
        Initializes the OMat24 Data Loader.
        :param base_path: Root directory containing the OMat24 tar file.
        :param split: The dataset split to load ('train', 'val', or 'test'). (NIFLHEIM does not have test)
        """

        hostname = socket.gethostname() # Will be applicable once we have data on Niflheim
        self.is_niflheim = "fysik.dtu.dk" in hostname or any(x in hostname for x in ["surt", "sylg", "slid", "svol"])
        if self.is_niflheim:
            self.base_path = "/home/scratch3/chipa/localDB/omat/OMat24_all/"
            if split == "test":
                logger.error("Niflheim does not have a test dataset!")
        else:
            self.base_path = base_path

        self.split = split
        self.split_dir = os.path.join(self.base_path, self.split)
        
        self.sub_folders = [
            "aimd-from-PBE-1000-npt", "aimd-from-PBE-1000-nvt",
            "aimd-from-PBE-3000-npt", "aimd-from-PBE-3000-nvt",
            "rattled-300-subsampled", "rattled-300",
            "rattled-500-subsampled", "rattled-500",
            "rattled-1000-subsampled", "rattled-1000",
            "rattled-relax"
        ]
        self.dataset = None

    def _prepare_data(self):
        """Checks if data is extracted and initializes dataset."""
        # (Extraction logic for local runs omitted for brevity)
        
        dataset_paths = []
        for sub in self.sub_folders:
            if self.is_niflheim:
                target_dir = os.path.join(self.split_dir, sub, sub)
            else:
                target_dir = os.path.join(self.split_dir, sub)
            found_files = glob.glob(os.path.join(target_dir, "*.aselmdb"))
            
            if found_files:
                dataset_paths.extend(found_files)
            else:
                logger.warning(f"No .aselmdb files found in: {target_dir}")

        try:
            if not dataset_paths:
                raise FileNotFoundError(f"No .aselmdb files found for split: {self.split}")
                
            # Fairchem's AseDBDataset handles a list of multiple file paths automatically
            self.dataset = AseDBDataset(config={"src": dataset_paths})
            logger.info(f"Loaded {len(dataset_paths)} files for split {self.split}")
        except Exception as e:
            logger.error(f"Error initializing AseDBDataset: {e}")
            raise

    def get_omat24(self, sample_size=None):
        """
        Processes the OMat24 dataset and returns a Pandas DataFrame and a Symbol Counter.
        :param sample_size: Integer. If provided, only processes the first N entries.
        """
        if self.dataset is None:
            self._prepare_data()

        total_to_process = sample_size if sample_size else len(self.dataset)
        logger.info(f"Beginning data processing for {total_to_process} entries.")
        
        data_list = []
        symbol_counts = Counter()

        for i in tqdm(range(total_to_process), desc=f"Extracting OMat24 {self.split} Properties"):
            try:
                atoms = self.dataset.get_atoms(i)
                info = atoms.info
                
                # Update symbol counts
                symbols = atoms.get_chemical_symbols()
                symbol_counts.update(symbols)
                
                # Extracting specific OMat24 metadata from your script
                entry = {
                    "index": i,
                    "atomic_numbers": atoms.get_atomic_numbers().tolist(),
                    "symbols": symbols,
                    "num_atoms": len(atoms),
                    "calc_id": info.get("calc_id"),
                    "task_type": info.get("task_type"),
                    "prototype_label": info.get("prototype_label"),
                    "energy_corrected_mp2020": info.get("energy_corrected_mp2020"),
                }
                
                data_list.append(entry)
            except Exception as e:
                logger.warning(f"Error processing entry at index {i}: {e}")

        logger.info("Data extraction finished. Creating DataFrame.")
        df = pd.DataFrame(data_list)
        
        logger.info(f"DataFrame created with {len(df)} rows.")
        logger.info(f"Detected elements: {list(symbol_counts.keys())}")
        
        return df, symbol_counts