import glob
import json
import logging
from collections import Counter
from typing import List, Tuple

import torch
from tokenizers import Tokenizer
from tokenizers.processors import BertProcessing
from torch.utils.data import Dataset
from tqdm import tqdm


class Bin2MLFunctionString(Dataset):
    """
    A torch dataset class for loading bin2ml generated ESIL strings
    """

    def __init__(
        self,
        data_dir: str,
        max_seq_len: int = 512,
        tokeniser_fp: Tuple[str, None] = None,
        dynamic_encoding: bool = True,
        with_labels: bool = True,
        filter_str=None,
        debug=False,
        remove_uniques=False,
    ):
        self.max_seq_len = max_seq_len
        self.max_seq_len_no_specials = max_seq_len - 2
        self.data_dir = data_dir if data_dir.endswith("/") else data_dir + "/"

        self.debug = debug
        self.filter_str = filter_str
        self.tokeniser_fp = tokeniser_fp
        self.dynamic_encoding = dynamic_encoding
        self.with_labels = with_labels
        self.remove_uniques = remove_uniques

        if self.debug:
            logging.basicConfig(level=logging.DEBUG)
        else:
            logging.basicConfig(level=logging.INFO)

        self.filepaths = self.get_data_filepaths()
        self.load_and_process_filepaths()
        if tokeniser_fp is not None:
            self.tokeniser = Tokenizer.from_file(self.tokeniser_fp)
            self.tokeniser.post_processor = BertProcessing(
                ("[SEP]", self.tokeniser.token_to_id("[SEP]")),
                ("[CLS]", self.tokeniser.token_to_id("[CLS]")),
            )
        if self.dynamic_encoding is False:
            self.tokenise_data_in_func_obj()

    def __len__(self):
        return len(self.function_objs)

    def __getitem__(self, item):
        """
        Retrives a function_obj based on an index

        Args:
            item: The index

        Returns:
            A function object

        Note:
            Depending on whether dynamic encoding has been set,
            this function will encode a function prior to being
            returned. Useful if pre-encoding examples (due to size/volume)
            is a pain
        """
        if self.dynamic_encoding and self.tokeniser_fp:
            encoding = self.tokeniser.encode(
                self.function_objs[item]["data"],
            )
            output = {
                "ids": encoding.ids,
                "attention_mask": encoding.attention_mask,
                "type_ids": encoding.type_ids,
                "label": self.function_objs[item]["label"],
            }
            return {key: torch.tensor(value) for key, value in output.items()}

        return self.function_objs[item]

    def tokenise_data_in_func_obj(self):
        """
        Tokenise all loaded function objects

        Returns:
            Updates self.function_objs in place with the tokenised equlivant
        """
        for i, ele in enumerate(self.function_objs):
            self.function_objs[i]["data"] = self.tokeniser.encode(
                self.function_objs[i]["data"], add_special_tokens=True
            )

    def get_data_filepaths(self) -> List[str]:
        """
        Get all data filepaths contained with self.data_dir.

        This can optionally be filtered by a filter string

        Returns:
            A list of filepaths
        """
        filepaths = glob.glob(self.data_dir + "*.json")
        if self.filter_str is not None:
            filepaths = [x for x in filepaths if self.filter_str in x]

        return filepaths

    def load_and_process_filepaths(self):
        """
        Loads and processes fileppaths

        This function does a lot of heavy lifting.

        It loads all filepaths to create function_objects and
        generates labels for the functions.

        Returns:
            A self.function_obj member variable full of function strings
        """
        self.function_objs = []
        self.unique_binary_funcs = []
        self.labels = []
        self.total_funcs = 0

        for file in tqdm(self.filepaths):
            binary_specific_unique_funcs = []

            logging.debug(f"Processing {file}..")
            binary_name = file.split("/")[-1].split("-")[0]

            # Load data
            fd = open(file, "r")
            data = json.load(fd)

            self.total_funcs += len(data)
            logging.debug(binary_name)
            logging.debug(f"All functions: {self.total_funcs}")

            [
                binary_specific_unique_funcs.append(f"{binary_name}{x['name']}")
                for x in tqdm(data)
                if f"{binary_name}{x['name']}" not in binary_specific_unique_funcs
            ]
            self.unique_binary_funcs.extend(binary_specific_unique_funcs)
            logging.debug(f"Only Uniques: {len(self.unique_binary_funcs)}")

            if self.with_labels:
                # Add labels to function objects
                logging.debug("Appending labels to function objs...")
                for function in tqdm(data):
                    label = self.unique_binary_funcs.index(
                        f"{binary_name}{function['name']}"
                    )
                    function["label"] = label

                    self.labels.append(label)

            self.function_objs.extend(data)

        if self.remove_uniques:
            logging.info("Removing Unique Functions from Dataset")
            label_counter = [k for k, v in Counter(self.labels).items() if v > 1]

            self.function_objs = [
                x for x in tqdm(self.function_objs) if x["label"] in label_counter
            ]
            logging.info("Rebuilding list index")
            self.labels = [x["label"] for x in tqdm(self.function_objs)]

        if self.with_labels:
            self.max_label = max(self.labels)
            self.label_counter = Counter(self.labels)
            logging.info("Sorting Labels...")
            self.labels = sorted(self.labels)
            logging.info("Sorting function objs...")
            self.function_objs = sorted(self.function_objs, key=lambda d: d["label"])

        logging.info("Dataset creation completed...")
