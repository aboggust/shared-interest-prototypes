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
class SaliencyImage(BaseModel):
    image: str
    bbox: list
    saliency: list
    label: str
    prediction: str
    score: str
    iou: str
    ground_truth_coverage: str
    explanation_coverage: str


class Bins(BaseModel):
    x0: float
    x1: float
    num: int


# Load case study datasets
datasets = ['data_dogs', 'data_vehicle', 'data_melanoma']
dataframes = {}
for dataset in datasets:
    dataframe = pd.read_json("./data/examples/%s.json" % dataset)
    dataframes[dataset] = dataframe.set_index('fname')


@app.get("/api/get-images", response_model=List[str])
async def get_images(case_study: str, method: str, sort_by: int,
                     prediction_fn: str, score_fn: str, label_filter: str,
                     iou_min: float, iou_max: float, ec_min: float,
                     ec_max: float, gtc_min: float, gtc_max: float):
    """ Get images from dataset given the current filters.

    Args:
        case_study: The name of the case study dataset.
        method: The name of the saliency method.
        sort_by: 1 if ascending, -1 if descending.
        prediction_fn: The prediction function. It can be 'all_images',
                       'correct_only', 'incorrect_only', or any label.
        score_fn: The score function name to apply.
        label_filter: The label filter to apply. It can be any label name or ''
                      for all labels.
        iou_min: Min iou score to keep.
        iou_max: Max iou score to keep.
        ec_min: Min explanation coverage score to keep.
        ec_max: Max explanation coverage score to keep.
        gtc_min: Min ground truth coverage score to keep.
        gtc_max: Max ground truth coverage score to keep.

    Returns:
        A list of image IDs from case_study filtered given the prediction_fn and
         label_filter and sorted by the score_fn in sort_by order.
    """
    df = dataframes[case_study]

    # Filter by prediction
    if prediction_fn == "all_images":
        pred_inds = np.ones(len(df))
    elif prediction_fn == "correct_only":
        pred_inds = df.label == df.prediction
    elif prediction_fn == "incorrect_only":
        pred_inds = df.label != df.prediction
    else:  # Assume predictionFn is a label
        pred_inds = df.prediction == prediction_fn

    # Filter by label 
    if label_filter == '':
        label_inds = np.ones(len(df))
    else:
        label_inds = df.label == label_filter

    # Filter by scores
    iou_inds = np.logical_and(df.iou.round(2) >= iou_min, df.iou.round(2) <= iou_max)
    ec_inds = np.logical_and(df.explanation_coverage.round(2) >= ec_min, df.explanation_coverage.round(2) <= ec_max)
    gtc_inds = np.logical_and(df.ground_truth_coverage.round(2) >= gtc_min, df.ground_truth_coverage.round(2) <= gtc_max)

    # Filter data frame.
    mask = np.logical_and.reduce((pred_inds, label_inds, iou_inds, ec_inds, gtc_inds))
    filtered_df = df.loc[mask].sort_values(score_fn, kind="mergesort",
                                           ascending=sort_by == 1)
    image_ids = list(filtered_df.index)
    return image_ids


@app.get("/api/get-saliency-image", response_model=SaliencyImage)
async def get_saliency_image(case_study: str, method: str, image_id: str,
                             score_fn: str):
    """Gets a single saliency image.

    Args:
        case_study: The name of the case study dataset.
        method: The name of the saliency method.
        image_id: The id of the image to return.
        score_fn: The score function to return.

    Returns:
        A dictionary of the image data for image_id from case_study. The 'score'
         key is set to the score_fn value.
    """
    df = dataframes[case_study]
    filtered_df = df.loc[image_id]
    filtered_df['score'] = filtered_df[score_fn]
    filtered_df['saliency'] = filtered_df[method]
    return filtered_df.to_dict()


@app.post("/api/get-saliency-images", response_model=List[SaliencyImage])
async def get_saliency_images(payload: api.ImagesPayload):
    """Gets saliency images.

        Args:
            payload: The payload containing the name of the case study, method,
                     image IDs, and the score function.

        Returns:
            A dictionary of the image data for the image IDs and case study in
            the payload. The 'score' key is the value of the score function in
            the payload.
        """
    payload = api.ImagesPayload(**payload)
    df = dataframes[payload.case_study]
    filtered_df = df.loc[payload.image_ids]
    filtered_df['score'] = filtered_df[payload.score_fn]
    return filtered_df.to_dict('records')


@app.get("/api/get-labels", response_model=List[str])
async def get_labels(case_study: str):
    """Gets the label values given the case study."""
    df = dataframes[case_study]
    return list(df.label.unique())


@app.get("/api/get-predictions", response_model=List[str])
async def get_predictions(case_study: str):
    """Gets the possible prediction values given the case study."""
    df = dataframes[case_study]
    return list(df.prediction.unique())


@app.post("/api/bin-scores", response_model=List[Bins])
async def bin_scores(payload: api.ImagesPayload, min_range: int = 0,
                     max_range: int = 1, num_bins: int = 11):
    """Bins the scores of the images.

    Args:
        payload: The payload containing the case study, method, images,
                 and score fn.
        min_range: The start of the bin range, inclusive. Defaults to 0.
        max_range: The end of the bin range, inclusive. Defaults to 1.
        num_bins: The number of bins to create. Defaults to 11.

    Returns:
        A list of dictionary objects containing the start, end, and number of
        scores in each bin.
    """
    payload = api.ImagesPayload(**payload)
    df = dataframes[payload.case_study]
    filtered_df = df.loc[payload.image_ids]
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
