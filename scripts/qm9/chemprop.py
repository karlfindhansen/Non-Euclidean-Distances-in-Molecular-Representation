# chemeleon_fingerprint.py
#
# this file contains the class CheMeleonFingerprint which can be instantiated
# and called to generate the CheMeleon learned embeddings for a list of SMILES
# strings and/or RDKit Mols. you may wish to simply copy or download this file directly for use,
# or adapt the code for your own purposes. No other files are required for it
# to work, though you must `pip install 'chemprop>=2.2.0'` for this to run.
#
# run `python chemeleon_fingerprint.py` for a quick usage demo, otherwise you
# should `import` the CheMeleonFingerprint class into your other code and use
# it there (following the example at the bottom of this file) to generate
# your learned fingerprints
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import torch
from chemprop import featurizers, nn
from chemprop.data import BatchMolGraph
from chemprop.models import MPNN
from chemprop.nn import RegressionFFN
from rdkit.Chem import Mol, MolFromSmiles
from loguru import logger


class CheMeleonFingerprint:
    def __init__(self, device: str | torch.device | None = None):
        logger.info("Initializing CheMeleonFingerprint generator...")
        self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        agg = nn.MeanAggregation()
        
        ckpt_dir = Path().home() / ".chemprop"
        ckpt_dir.mkdir(exist_ok=True)
        mp_path = ckpt_dir / "chemeleon_mp.pt"
        
        if not mp_path.exists():
            logger.warning(f"CheMeleon weights not found at {mp_path}. Initiating download from Zenodo...")
            try:
                urlretrieve(
                    r"https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    mp_path,
                )
                logger.success(f"Successfully downloaded weights to {mp_path}")
            except Exception as e:
                logger.error(f"Failed to download CheMeleon weights: {e}")
                raise
        else:
            logger.info(f"Found existing CheMeleon weights at {mp_path}")

        try:
            chemeleon_mp = torch.load(mp_path, weights_only=True)
            mp = nn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
            mp.load_state_dict(chemeleon_mp["state_dict"])
            logger.success("Successfully loaded CheMeleon message passing weights.")
        except Exception as e:
            logger.error(f"Failed to load weights from {mp_path}. The file might be corrupted: {e}")
            raise
            
        self.model = MPNN(
            message_passing=mp,
            agg=agg,
            predictor=RegressionFFN(input_dim=mp.output_dim),  # not actually used
        )
        self.model.eval()
        
        if device is not None:
            try:
                self.model.to(device=device)
                logger.info(f"CheMeleon model moved to device: {device}")
            except Exception as e:
                logger.error(f"Failed to move model to device '{device}': {e}")
                raise

    def __call__(self, molecules: list[str | Mol]) -> np.ndarray:
        logger.info(f"Generating CheMeleon fingerprints for {len(molecules)} molecules...")
        
        valid_mols = []
        for idx, m in enumerate(molecules):
            mol = MolFromSmiles(m) if isinstance(m, str) else m
            if mol is None:
                logger.warning(f"Molecule at index {idx} (input: {m}) could not be parsed and will be skipped.")
            else:
                valid_mols.append(mol)
                
        if not valid_mols:
            logger.error("No valid molecules provided for fingerprint generation.")
            return np.array([])

        try:
            bmg = BatchMolGraph([self.featurizer(m) for m in valid_mols])
        except Exception as e:
            logger.error(f"Failed to featurize molecules into BatchMolGraph: {e}")
            raise
            
        bmg.to(device=self.model.device)
        
        with torch.no_grad():
            try:
                fps = self.model.fingerprint(bmg).numpy(force=True)
                logger.success(f"Successfully generated {fps.shape[0]} fingerprints of dimension {fps.shape[1]}.")
                return fps
            except Exception as e:
                logger.error(f"Error during model forward pass (fingerprint extraction): {e}")
                raise


if __name__ == "__main__":
    logger.info("Starting quick usage demo for CheMeleonFingerprint...")
    try:
        chemeleon_fingerprint = CheMeleonFingerprint()
        fps = chemeleon_fingerprint(["C", "CC", MolFromSmiles("CCC"), "INVALID_SMILES"])
        if fps.size > 0:
            print(f"Sample fingerprint dimension: {len(fps[0])}")
            print("Demo completed successfully.")
    except Exception as e:
        logger.error(f"Demo failed: {e}")