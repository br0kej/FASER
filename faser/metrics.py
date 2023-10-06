from typing import Dict

import torch
from torchmetrics.retrieval import (
    RetrievalMRR,
    RetrievalNormalizedDCG,
    RetrievalPrecision,
    RetrievalRecall,
)


def eval_model_no_model(
    indexes: torch.LongTensor,
    preds: torch.Tensor,
    target: torch.Tensor,
) -> Dict:
    """
    Evaluate a models outputs and produce a metric dictionary

    The following metrics are computed by default at 3 K thresholds (1, 5, 10):
        - Retrieval Recall
        - Retrieval Precision
        - Normalized Discounted Cumulative Gain (NDCG)

    Args:
        indexes: A torch.LongTensor containing an index of which preds/targets are related to which query
        preds: A torch.Tensor containing the similarity scores outputted by a given model
        target: A torch.Tensor containing bools denoting whether a return item is relevant to the query or not
        print_stats: A bool flag to toggle printing the metric dictionary

    Returns:
        metric dict: A dictionary containing the calculated metrics where the keys are the metric name and values
                     the calculated metrics
    """
    k_intervals = [1, 5, 10]
    metric_dict = {}

    metric_name_func_tuples = [("MRR", RetrievalMRR())]
    metric_names = ["MRR@", "R@", "P@", "NDCG@"]
    metric_funcs = [
        RetrievalMRR,
        RetrievalRecall,
        RetrievalPrecision,
        RetrievalNormalizedDCG,
    ]

    for metric_name, metric_func in zip(metric_names, metric_funcs):
        for interval in k_intervals:
            metric_name_func_tuples.append(
                (f"{metric_name}{interval}", metric_func(top_k=interval))
            )

    for name, func in metric_name_func_tuples:
        metric_dict[name] = func(preds, target, indexes=indexes).item()

    metric_dict["highest_sim"] = max(preds)
    metric_dict["lowest_sim"] = min(preds)

    return metric_dict
