# About
This is a class project for a course in Digital Image Processing. It combines a vision backbone with a large language model to alert a driver of potential incidents.

# Installation
This project is purely written in Python and its dependencies can be installed with either pip or pixi. It is recommended to use a newer version of python, such as 3.14.

## Pip
To install the project's dependencies using pip, run the following commands

```bash
python -m venv .venv
source .venv/bin/activate # or the equivalent on Windows
pip install -r requirements.txt
pip install -e . # install the project as a module
```


## Pxix Installation
If using [pixi](https://pixi.prefix.dev/latest/), you can install the project and its dependencies with `pixi install` and then `pixi run pip install -e .`.



# Data Collection
Once the dependencies have been installed, a python script can be ran to download a few videos to run on the pipeline. As an example, running

```bash
python -m driving_assistant.data_utils.reddit_scraper --subreddit=MildlyBadDrivers --limit=5
```

will attempt to download 5 videos from the subreddit MildlyBadDrivers, and save them in a directory called `data/reddit_dashcam_videos`.


# Running the pipeline
After downloading a few videos, the pipeline can be ran using

```bash
python main.py --visualize --llm-model=Qwen/Qwen2.5-1.5B-Instruct --object-model=PekingU/rtdetr_r50vd --output-dir=eval_videos
```
