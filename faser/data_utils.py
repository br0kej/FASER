import os

from faser.dataset import Bin2MLFunctionString


def fstrs_to_tokeniser_training_data(data_dir: str, output_name: str) -> None:
    """
    Converts functions string data objects into single newline delimited data usable
    to train a tokeniser with and write to a file

    Args:
        data_dir: The path to a corpus of function string JSON objects
        output_name: The name of the output file

    Returns:
        A .txt file containing newline delimited function strings
    """
    if os.path.exists(data_dir):
        data = Bin2MLFunctionString(data_dir, with_labels=False)

        with open(output_name, "w") as fileout:
            for i, ele in enumerate(data):
                fileout.write(ele["data"] + "\n")
    else:
        raise ValueError(f"{data_dir} does not exist. Enter a valid path.")


if __name__ == "__main__":
    fstrs_to_tokeniser_training_data("data-samples/", "for_tokenizer.txt")
