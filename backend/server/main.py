import argparse
import os
from typing import *

import numpy as np
import pandas as pd
import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, RedirectResponse
from pydantic import BaseModel

import backend.server.api as api
import backend.server.path_fixes as pf

parser = argparse.ArgumentParser(
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--port", default=5050, type=int,
                    help="Port to run the app. ")

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

prefix = os.environ.get('CLIENT_PREFIX', 'client')


# Main routes
@app.get("/")
def index():
    return RedirectResponse(url=f"{prefix}/index.html")


# the `file_path:path` says to accept any path as a string here.
# Otherwise, `file_paths` containing `/` will not be served properly
@app.get("/client/{file_path:path}")
def send_static_client(file_path: str):
    """ Serves all files from ./client/ to ``/client/{path}``.
    Used primarily for development. NGINX handles production.

    Args:
        file_path: Name of file in the client directory
    """
    f = str(pf.DIST / file_path)
    print("Finding file: ", f)
    return FileResponse(f)


# ======================================================================
# MAIN API
# ======================================================================
class SaliencyText(BaseModel):
    words: list
    saliency_inds: list
    ground_truth_inds: list
    label: str
    prediction: float # TODO update for non-float predictions
    iou: float
    ground_truth_coverage: float
    saliency_coverage: float


class Bins(BaseModel):
    x0: float
    x1: float
    num: int


# Load datasets
saliency_methods = ['sis', 'lime', 'integrated_gradients']
aspects = ['aspect0', 'aspect1', 'aspect2']
base_datasets = ['data_beeradvocate',]
datasets = ['%s_%s_%s' %(dataset, method, aspect)
            for dataset in base_datasets
            for method in saliency_methods
            for aspect in aspects]
dataframes = {}
for dataset in datasets:
    dataframe = pd.read_json("./data/examples/%s.json" % dataset)
    dataframes[dataset] = dataframe.set_index('name')


@app.get("/api/get-result-ids", response_model=List[str])
async def get_result_ids(dataset: str, method: str, sort_by: int,
                         prediction_fn: str, score_fn: str, label_filter: str,
                         iou_min: float, iou_max: float, sc_min: float,
                         sc_max: float, gtc_min: float, gtc_max: float):
    """ Get results from dataset given the current filters.
    Args:
        dataset: The name of the dataset.
        method: The name of the saliency method.
        sort_by: 1 if ascending, -1 if descending.
        prediction_fn: The prediction function. It can be 'all',
                       'correct_only', 'incorrect_only', or any label.
        score_fn: The score function name to apply.
        label_filter: The label filter to apply. It can be any label name or ''
                      for all labels.
        iou_min: Min iou score to keep.
        iou_max: Max iou score to keep.
        sc_min: Min saliency coverage score to keep.
        sc_max: Max saliency coverage score to keep.
        gtc_min: Min ground truth coverage score to keep.
        gtc_max: Max ground truth coverage score to keep.
    Returns:
        A list of image IDs from dataset filtered given the prediction_fn and
         label_filter and sorted by the score_fn in sort_by order.
    """
    dataset_name = 'data_beeradvocate_%s_%s' % (method, dataset)
    df = dataframes[dataset_name]

    # Handle regression data
    is_prediction_correct = lambda df: df.label == df.prediction
    is_prediction_equal = lambda df, prediction: df.prediction == prediction
    delta = 0.05
    if df.prediction.dtype == 'float64':
        def is_prediction_correct(df):
            return np.logical_and(
                df.prediction >= pd.to_numeric(df.label) - delta,
                df.prediction <= pd.to_numeric(df.label) + delta
            )

        def is_prediction_equal(df, prediction):
            min_value, max_value = [float(value) for value in prediction.split('-')]
            return np.logical_and(df.prediction >= min_value, df.prediction <= max_value)

    # Filter by prediction
    if prediction_fn == "all":
        prediction_indices = np.ones(len(df))
    elif prediction_fn == "correct_only":
        prediction_indices = is_prediction_correct(df)
    elif prediction_fn == "incorrect_only":
        prediction_indices = ~is_prediction_correct(df)
    else:  # Assume predictionFn is a label
        prediction_indices = is_prediction_equal(df, prediction_fn)

    # Filter by label
    if label_filter == '':
        label_indices = np.ones(len(df))
    else:
        label_indices = df.label.apply(str) == label_filter

    # Filter by scores
    iou_indices = np.logical_and(
        df.iou.round(2) >= iou_min,
        df.iou.round(2) <= iou_max)
    sc_indices = np.logical_and(
        df.saliency_coverage.round(2) >= sc_min,
        df.saliency_coverage.round(2) <= sc_max)
    gtc_indices = np.logical_and(
        df.ground_truth_coverage.round(2) >= gtc_min,
        df.ground_truth_coverage.round(2) <= gtc_max)

    # Filter data frame.
    mask = np.logical_and.reduce((prediction_indices, label_indices,
                                  iou_indices, sc_indices, gtc_indices))
    filtered_df = df.loc[mask].sort_values(score_fn,
                                           kind="mergesort",
                                           ascending=sort_by == 1)
    image_ids = list(filtered_df.index)
    return image_ids


@app.get("/api/get-result", response_model=SaliencyText)
async def get_result(dataset: str, method: str, result_id: str):
    """Gets a single saliency result.
    Args:
        dataset: The name of the dataset.
        method: The name of the saliency method.
        result_id: The id of the result to return.
    Returns:
        A dictionary of the result data for result_id from dataset.
    """
    dataset_name = 'data_beeradvocate_%s_%s' %(method, dataset)
    df = dataframes[dataset_name]
    filtered_df = df.loc[int(result_id)]
    return filtered_df.to_dict()


@app.get("/api/get-labels", response_model=List[str])
async def get_labels(dataset: str, method: str):
    """Gets the label values given the dataset."""
    dataset_name = 'data_beeradvocate_%s_%s' %(method, dataset)
    df = dataframes[dataset_name]
    return sorted(list(df.label.unique()))


@app.get("/api/get-predictions", response_model=List[str])
async def get_predictions(dataset: str, method: str, delta: float=0.1):
    """Gets the possible prediction values given the dataset and method."""
    dataset_name = 'data_beeradvocate_%s_%s' %(method, dataset)
    df = dataframes[dataset_name]

    # Handle regression data
    if df.prediction.dtype == 'float64':
        min_label, max_label = 0.0, pd.to_numeric(df.label).max()
        arr = np.arange(min_label, max_label, delta)
        ranges = [(arr[i-1], arr[i]) for i in range(1, len(arr))]
        predictions = ['%.1f-%.1f' % x for x in ranges]
        return predictions
    return list(df.prediction.unique())


@app.post("/api/bin-scores", response_model=List[Bins])
async def bin_scores(payload: api.ResultPayload, min_range: int = 0,
                     max_range: int = 1, num_bins: int = 11):
    """Bins the scores of the results.

    Args:
        payload: The payload containing the dataset, method, results, and score fn.
        min_range: The start of the bin range, inclusive. Defaults to 0.
        max_range: The end of the bin range, inclusive. Defaults to 1.
        num_bins: The number of bins to create. Defaults to 11.

    Returns:
        A list of dictionary objects containing the start, end, and number of
        scores in each bin.
    """
    payload = api.ResultPayload(**payload)
    dataset_name = 'data_beeradvocate_%s_%s' % (payload.method, payload.dataset)
    df = dataframes[dataset_name]
    filtered_df = df.loc[[int(id) for id in payload.result_ids]]
    scores = filtered_df[payload.score_fn].tolist()
    bins = np.linspace(min_range, max_range, num_bins)
    hist, bin_edges = np.histogram(scores, bins)
    bin_object = [{'x0': bin_edges[i], 'x1': bin_edges[i + 1], 'num': num}
                  for i, num in enumerate(list(hist))]
    return bin_object


if __name__ == "__main__":
    # This file is not run as __main__ in the uvicorn environment
    args, _ = parser.parse_known_args()
    uvicorn.run("server:app", host='127.0.0.1', port=args.port)
