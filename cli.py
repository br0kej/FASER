import argparse
import logging

from tokenizers import Tokenizer

from faser.data_utils import fstrs_to_tokeniser_training_data
from faser.eval import FaserGeneralFuncSearchEval, FaserVulnSearchEval
from faser.tokeniser import print_tokeniser_encode_example, train_bpe_tokeniser
from faser.train import FASERTrain

parser = argparse.ArgumentParser(
    prog="cli.py", description="Frontend for training and testing FASER models"
)

sub_parsers = parser.add_subparsers(dest="command")
sub_parsers.required = True

# Creating Tokenisers
parser_tokeniser = sub_parsers.add_parser(
    "tokeniser", help="train a huggingface tokeniser"
)
parser_tokeniser.add_argument(
    "-o", "--output-path", required=True, help="the name of the tokeniser"
)
parser_tokeniser.add_argument(
    "-i",
    "--input-data",
    nargs="+",
    required=True,
    help="the input data corpus to train the tokeniser",
)

parser_tokeniser.add_argument(
    "-m",
    "--max-seq-len",
    required=True,
    help="the max seq length for the tokeniser to make (including specials)",
)

# Testing Tokeniser/Encode
parser_encode = sub_parsers.add_parser(
    "encode", help="encode a string with a given tokeniser"
)
parser_encode.add_argument("input", type=str, help="the input sequence to encode")
parser_encode.add_argument("-t", "--tokeniser-path", help="path to a trained tokeniser")

# Convert function as string data to a text file
parser_fstr2txt = sub_parsers.add_parser(
    "fstr2txt",
    help="convert fstr JSON to newline delimited text file to train tokeniser",
)
parser_fstr2txt.add_argument("data_dir", help="path to data directory")
parser_fstr2txt.add_argument("-o", "--output-name", help="output name", required=True)

# Training a FASER model
parser_train = sub_parsers.add_parser("train", help="Train a FASER model")
parser_train.add_argument("--train_data", help="path to traini data", required=True)
parser_train.add_argument("--test_data", help="path to test data", required=True)
parser_train.add_argument(
    "-t", "--tokeniser_fp", help="path to tokeniser", required=True
)
parser_train.add_argument("-n", "--name", help="name of model", required=True)

## Good Defaults
parser_train.add_argument("-e", "--epochs", help="Num training epochs", default=30)
parser_train.add_argument("-b", "--batch_size", help="Training batch size", default=8)
parser_train.add_argument(
    "-lr", "--learning_rate", help="Training learning rate", default=0.00005
)
parser_train.add_argument(
    "--num_accumlation_steps", help="Steps to do gradient accumlation too", default=512
)
parser_train.add_argument(
    "--gradient_accumulation",
    help="toggle to set gradient accumlation",
    action="store_false",
)
parser_train.add_argument("--num_pos_pairs_in_batch", default=2)
parser_train.add_argument("--filter-str", default=None)

# Evaluating FASER Model - General Function Search
parser_func_search_eval = sub_parsers.add_parser(
    "fseval", help="General function search evaluation for FASER"
)

parser_func_search_eval.add_argument(
    "-n",
    "--num-eval-sp",
    type=int,
    help="Number of Search Pools to eval with",
    default=1400,
)
parser_func_search_eval.add_argument(
    "-i", "--input-dim", type=int, help="Size of input dimension", default=4096
)
parser_func_search_eval.add_argument(
    "-d", "--eval-data", type=str, help="Path to the evaluation data", required=True
)
parser_func_search_eval.add_argument(
    "-t", "--tokeniser", type=str, help="Path to the tokeniser", required=True
)
parser_func_search_eval.add_argument(
    "-m", "--model", type=str, help="Path to the model", required=True
)
parser_func_search_eval.add_argument(
    "-s",
    "--sp-size",
    type=int,
    help="Number of elements in a single search pool",
    default=100,
)
parser_func_search_eval.add_argument(
    "-f",
    "--filter-str",
    type=str,
    help="Filter which training filenames to use",
    required=False,
)

# Evaluating FASER Model - Vulnerability Function Search
parser_func_search_eval.add_argument("-v", "--verbose", action="store_true")

parser_vuln_search_eval = sub_parsers.add_parser(
    "vseval", help="Vulnerability search evaluation for FASER"
)
parser_vuln_search_eval.add_argument(
    "-i", "--input-dim", type=int, help="Size of input dimension", default=4096
)
parser_vuln_search_eval.add_argument(
    "-d", "--eval-data", type=str, help="Path to the evaluation data", required=True
)
parser_vuln_search_eval.add_argument(
    "-t", "--tokeniser", type=str, help="Path to the tokeniser", required=True
)
parser_vuln_search_eval.add_argument(
    "-m", "--model", type=str, help="Path to the model", required=True
)
parser_vuln_search_eval.add_argument(
    "-f",
    "--model-friendly-name",
    type=str,
    help="Model name for output artefacts",
    required=True,
)
args = parser.parse_args()

if args.command == "fstr2txt":
    fstrs_to_tokeniser_training_data(args.data_dir, args.output_name)

elif args.command == "tokeniser":
    tokeniser = train_bpe_tokeniser(
        args.input_data, max_seq_length=int(args.max_seq_len)
    )
    tokeniser.save(f"{args.max_seq_len}-{args.output_path}")

elif args.command == "encode":
    tokeniser = Tokenizer.from_file(args.tokeniser_path)
    print_tokeniser_encode_example(tokeniser, args.input)

elif args.command == "train":
    model = FASERTrain(
        name=args.name,
        train_data_fp=args.train_data,
        test_data_fp=args.test_data,
        tokeniser_fp=args.tokeniser_fp,
        batch_size=args.batch_size,
        num_pos_pairs_in_batch=args.num_pos_pairs_in_batch,
        learning_rate=args.learning_rate,
        num_training_epochs=args.epochs,
        num_accumlation_steps=args.num_accumlation_steps,
        filter_str=args.filter_str,
    )

    model.train()

elif args.command == "fseval":
    if args.verbose:
        logging.basicConfig(level=logging.DEBUG)

    eval_run = FaserGeneralFuncSearchEval(
        torch_model_bin_fp=args.model,
        eval_data_fp=args.eval_data,
        max_seq_len=args.input_dim,
        tokeniser_fp=args.tokeniser,
        filter_str=args.filter_str,
        num_eval_search_pools=args.num_eval_sp,
        search_pool_size=args.sp_size,
    )

    eval_run.eval()

elif args.command == "vseval":
    eval_run = FaserVulnSearchEval(
        torch_model_bin_fp=args.model,
        eval_data_dir=args.eval_data,
        max_seq_len=args.input_dim,
        tokeniser_fp=args.tokeniser,
        model_friendly_name=args.model_friendly_name,
    )

    eval_run.rank()
