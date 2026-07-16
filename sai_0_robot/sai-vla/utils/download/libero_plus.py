from pathlib import Path

from datasets import load_dataset
from datasets.exceptions import NonMatchingSplitsSizesError

output_dir = Path("/data_disk1/hwl/libero_plus")
output_dir.mkdir(parents=True, exist_ok=True)

try:
    ds = load_dataset("Sylvest/libero_plus_lerobot")
except NonMatchingSplitsSizesError:
    # Some Hub datasets ship stale split metadata. Fall back to
    # disabling split-size verification so the dataset can still load.
    ds = load_dataset(
        "Sylvest/libero_plus_lerobot",
        verification_mode="no_checks",
    )

ds.save_to_disk(str(output_dir))

