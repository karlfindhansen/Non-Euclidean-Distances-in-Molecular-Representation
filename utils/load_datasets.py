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
from mp_api.client import MPRester

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

        total_to_process = sample_size if sample_size and sample_size < len(self.dataset) else len(self.dataset)
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
    def __init__(self, base_path="data/Materials Project/", file_name="comprehensive_mp_dataset.csv", api_key="XbTAm6OndiM15hj8OErnTmKHxHopX0AE"):
        """
        Initializes the Materials Project Loader.
        :param base_path: Directory to store/load the CSV.
        :param file_name: The name of the CSV file.
        :param api_key: Your Materials Project API Key.
        """
        self.api_key = api_key
        self.base_path = base_path
        self.file_name = file_name
        self.file_path = os.path.join(self.base_path, self.file_name)
        self.df = pd.DataFrame()

        hostname = socket.gethostname()
        self.is_niflheim = "fysik.dtu.dk" in hostname or any(x in hostname for x in ["surt", "sylg", "slid", "svol"])
        
        if not os.path.exists(self.base_path):
            os.makedirs(self.base_path, exist_ok=True)

    def load_data(self, force_fetch=False):
        """
        Loads data from CSV if it exists, otherwise fetches from MPRester.
        :param force_fetch: If True, ignores the CSV and fetches fresh data from the API.
        """
        if os.path.exists(self.file_path) and not force_fetch:
            logger.info(f"Found existing dataset at {self.file_path}. Loading...")
            self.df = pd.read_csv(self.file_path)
        else:
            if force_fetch:
                logger.info("Force fetch requested. Connecting to Materials Project API...")
            else:
                logger.info(f"Dataset not found at {self.file_path}. Fetching from API...")

            try:
                with MPRester(self.api_key) as mpr:
                    # Get all available fields for comprehensiveness
                    comprehensive_fields = mpr.materials.summary.available_fields
                    
                    # Search for all materials
                    results = mpr.materials.summary.search(fields=comprehensive_fields)
                    
                    # Convert DataDoc objects to dictionaries
                    data_list = [doc.dict() for doc in results]
                    
                    self.df = pd.DataFrame(data_list)
                    
                    # Save to CSV for the "second time" call
                    self.df.to_csv(self.file_path, index=False)
                    logger.info(f"Dataset saved to {self.file_path}")
                    
            except Exception as e:
                logger.error(f"Failed to fetch data from Materials Project: {e}")
                raise

        logger.info(f"Successfully loaded {len(self.df)} entries with {len(self.df.columns)} properties.")
        return self.df

    def get_summary(self):
        if self.df.empty:
            logger.warning("No data loaded. Call load_data() first.")
            return
        print(f"\n--- Materials Project Dataset Summary ---")
        print(f"Total Materials: {len(self.df)}")
        print(f"Total Properties: {len(self.df.columns)}")
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

        total_to_process = sample_size if sample_size and sample_size < len(self.dataset) else len(self.dataset)
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