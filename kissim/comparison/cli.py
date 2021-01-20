"""
cli_similarity.py

Subpocket-based structural fingerprint for kinase pocket comparison.

Execute similarity step.
"""

import argparse
import pickle
import logging
from pathlib import Path

from kissim.comparison import Similarity


# Parse arguments
parser = argparse.ArgumentParser()

parser.add_argument(
    "-o", "--output", type=str, help="Path to output (results) folder", required=True
)
args = parser.parse_args()

# Parameters
PATH_RESULTS = Path(args.output)

distance_measures = {"scaledEuclidean": "scaled_euclidean", "scaledCityblock": "scaled_cityblock"}
feature_weighting_schemes = {
    "weights100": [1.0, 0.0, 0.0],
    "weights010": [0.0, 1.0, 0.0],
    "weights001": [0.0, 0.0, 1.0],
    "weights110": [0.5, 0.5, 0.0],
    "weights101": [0.5, 0.0, 0.5],
    "weights011": [0.0, 0.5, 0.5],
    "weights111": [1.0 / 3, 1.0 / 3, 1.0 / 3],
}

# Logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%m/%d/%Y %I:%M:%S %p",
    filename=PATH_RESULTS / "cli_similarity.log",
    filemode="w",
    level=logging.INFO,
)
console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger("").addHandler(console)


if __name__ == "__main__":

    logger.info(f"Loading FingerprintGenerator from disc...")
    with open(PATH_RESULTS / "encoding" / "fingerprint_generator.p", "rb") as f:
        fingerprint_generator = pickle.load(f)
    logger.info(f"Done.")

    for key, value in distance_measures.items():
        logger.info(f"Selected distance measure: {key}:  {value}")
    for key, value in feature_weighting_schemes.items():
        logger.info(f"Selected feature weighting schemes: {key}: {value}")

    similarity = Similarity()
    similarity.execute(
        fingerprint_generator, distance_measures, feature_weighting_schemes, PATH_RESULTS
    )
