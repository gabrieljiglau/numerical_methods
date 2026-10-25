import polars as pl
import argparse
from pathlib import Path


def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Split the input CSVs")
    p.add_argument(
        "--input_dir",
        type=str,
        default="input",
        help="Path to the folder containing the input csv (A, x and b)",
    )
    p.add_argument(
        "--matrix",
        type=str,
        default="A.csv",
        help="The CSV file for the input matrix",
    )
    p.add_argument(
        "--variables",
        type=str,
        default="x.csv",
        help="The CSV file for the input vector",
    )
    p.add_argument(
        "--target",
        type=str,
        default="b.csv",
        help="The CSV file for the target vector",
    )

    return p.parse_args()


def convert_input() -> tuple[None, ...]:

    """ converts the 3 input CSVs (A, x and b) to numpy arrays and returns them as a tuple
    """

    args = _parse_args()
    filenames = [args.matrix, args.variables, args.target]
    input_dir = Path(args.input_dir)

    converted = []
    for i, filename in enumerate(filenames):

        df = pl.read_csv(input_dir / filename, has_header=False)

        # all entries should be numbers
        all_numeric = all(dtype.is_numeric() for dtype in df.dtypes)
        if not all_numeric:
            raise ValueError(f"{filenames[i]} does not have all entries as numbers")

        arr = df.to_numpy()

        # if it's a 1D vector, 'to_numpy()' still returns a (N, 1) matrix
        # 'squeeze()' removes the last dimension
        converted.append(arr if i == 0 else arr.squeeze())

    converted_tuple = tuple(converted)

    # check if the matrix / vectors can be multiplied
    A = converted_tuple[0]
    x = converted_tuple[1]
    b = converted_tuple[2]

    if A.shape[1] != x.shape[0]:
        raise ValueError(f"The input matrix with the shape {A.shape} cannot be "
                         f"multiplied with input vector, which is of shape {x.shape}")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"The input matrix with {A.shape} is not compatible"
                         f"with the target , which is of shape {b.shape}")

    return converted_tuple


