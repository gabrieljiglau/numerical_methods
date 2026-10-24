import numpy as np
import polars as pl
import argparse


def _parse_args() -> argparse.Namespace:

    p = argparse.ArgumentParser(description="Split the input CSVs")
    p.add_argument(
        "--input-dir",
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


def convert_input(args: argparse.Namespace) -> tuple[None, ...]:

    """ converts the 3 input CSVs (A, x and b) to numpy arrays and returns them as a tuple
    """

    args = _parse_args()
    input_list = ["matrix", "variables", "target"]

    converted_tuple = []
    for i in range(len(input_list)):

        df = pl.read_csv(getattr(args, input_list[i]))

        # all entries should be numbers
        all_numeric = all(dtype.is_numeric() for dtype in df.dtypes)
        if not all_numeric:
            raise ValueError(f"{input_list[i]} does not have all entries as numbers")

        arr = df.to_numpy()

        # if it's a 1D vector, 'to_numpy()' still returns a (N, 1) matrix
        # 'squeeze()' removes the last dimension
        converted_tuple[i] = arr if i == 0 else arr.squeeze()

    converted_tuple = tuple(converted_tuple)

    # check if the matrix / vectors can be multiplied
    A = converted_tuple[0]
    x = converted_tuple[1]
    b = converted_tuple[2]

    if A.shape[1] != x.shape[0]:
        raise ValueError(f"The input matrix with the shape {A.shape} cannot be"
                         f"multiplied with input vector, which is of shape {x.shape}")
    if A.shape[0] != b.shape[0]:
        raise ValueError(f"The input matrix with {A.shape} is not compatible"
                         f"with the target , which is of shape {b.shape}")

    return converted_tuple


