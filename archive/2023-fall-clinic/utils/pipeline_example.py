# These first lines allow us to access code from sibling directories
from pathlib import Path

from utils.preprocess_util_lib_example import save_random_dataframe

current_directory = Path(__file__).parent
repo_root = current_directory.parent

if __name__ == "__main__":

    # This is an example of running the code as a pipeline
    # Rather than through a notebook
    output_directory = repo_root / "output"
    output_file = "sample_output.csv"
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    save_random_dataframe(
        output_directory,
        output_file,
    )
