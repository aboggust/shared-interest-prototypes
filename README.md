# Shared Interest ([Article](https://dl.acm.org/doi/pdf/10.1145/3491102.3501965) | [Demo](http://shared-interest.csail.mit.edu))
![Shared interest teaser](./client/src/assets/img/teaser.svg)

This repository contains the interface code for:

[Shared Interest: Measuring Human-AI Alignment to Identify Recurring Patterns in Model Behavior](https://arxiv.org/abs/2107.09234)  
Authors: [Angie Boggust](http://angieboggust.com/), [Benjamin Hoover](https://www.bhoov.com/), [Arvind Satyanarayan](https://arvindsatya.com/), and [Hendrik Strobelt](http://hendrik.strobelt.com/)

Shared Interest is a method to quantify model behavior by comparing human and model decision making. In Shared Interest, human decision is approximated via ground truth annotations and model decision making is approximated via saliency. By quantifying each instance in a dataset, Shared Interest can enable large-scale analysis of model behavior.

Each interface is stored in a seperate branch. The computer vision interface is on `computer-vision`, the NLP interface is on `nlp`, and the intreactive probing interface is on `interactive-probing`.

## Getting Started

Before cloning this repo, [install](https://docs.github.com/en/free-pro-team@latest/github/managing-large-files/installing-git-large-file-storage) `git lfs`. When you clone the repo, the data files will automatically download. 

From the root:

1. `conda env create -f environment.yml`
2. `conda activate shared-interest`
3. `pip install -e .`
2. `cd client; npm i; npm run build`

The main demo is available at `/`.

## Running Locally
To start the server for development, run:

`uvicorn backend.server:app --reload`

For production, run:

`uvicorn backend.server:app`

This will run on a single worker, which should be sufficient for this.
By default this will run on `127.0.0.1:8000`.
To change the host or the port, run:

`uvicorn backend.server:app --host <host> --port <port>`

## Creating Data Files
The code in `data/` is used to create the data files consumed by Shared Interest.
To apply it to your own data, models, and explanation methods, modify `data/generate_datasets.py` and `data/explanation_methods.py`.

Once you have created your own data file, you can incorporate it into the interface, by adding it to `backend/server/api/main.py`
and to the case study selection bar in `client/src/ts/etc/selectionOptions.ts`.
