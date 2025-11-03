from pathlib import Path
import folder_paths
import random
from PIL import Image
import torch
import numpy as np
from matplotlib.axes import Axes
from matplotlib.ticker import StrMethodFormatter
from typing import Optional


def safe_tempfile() -> Path:
    tempdir = Path(folder_paths.get_temp_directory())
    while (filepath := tempdir / f"{random.randint(1000000,9999999)}.png").exists(): pass
    return filepath

def load_image(filepath:Path|str) -> torch.Tensor:
    with Image.open(filepath) as img:
        img = img.convert("RGB")
        image:torch.Tensor = torch.from_numpy(np.array(img).astype(np.float32) / 255.0).unsqueeze(0)
    return image

def label_plot(ax:Axes, title:Optional[str]=None, xlabel:Optional[str]=None, ylabel:Optional[str]=None, xformat:Optional[str]=None, yformat:Optional[str]=None):
    if title:   ax.set_title(title)
    if xlabel:  ax.set_xlabel(xlabel)
    if ylabel:  ax.set_ylabel(ylabel)
    if xformat: ax.xaxis.set_major_formatter(StrMethodFormatter(xformat))
    if yformat: ax.yaxis.set_major_formatter(StrMethodFormatter(yformat))
    ax.legend()