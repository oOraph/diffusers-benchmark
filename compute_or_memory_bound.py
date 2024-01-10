import os
import time
import torch
import datasets
from diffusers import StableDiffusionPipeline
import numpy as np
import pandas as pd

os.environ['HF_HUB_ENABLE_HF_TRANSFER'] = '1'

height=512
width=512
num_inference_steps=50
batch_sizes = [1, 2, 4, 8, 12, 16, 20, 24 ]
outputs = '/tmp/results.csv'
np.random.seed(42)

diffusion = StableDiffusionPipeline.from_pretrained("runwayml/stable-diffusion-v1-5")
prompts = datasets.load_dataset("Gustavosta/Stable-Diffusion-Prompts", split='train')

df = pd.DataFrame([], columns=['inference_steps', 'batch', 'time', 'time_per_image'])

if torch.cuda.is_available():
    print("Moving diffusion model to GPU")
    diffusion.to('cuda')


for i, size in enumerate(batch_sizes):
    select_indices = np.random.choice(range(0, len(prompts)), size, replace=False)
    selected = prompts.select(select_indices)
    start = time.time()
    images = diffusion(
        prompt=[p['Prompt'] for p in selected],
        num_inference_steps=int(num_inference_steps),
        height=height,
        width=width
    ).images
    elapsed = time.time() - start
    print("Elapsed {:.2f} s".format(elapsed))
    time_per_image = elapsed / size
    print("Elapsed per image {:2f} s".format(time_per_image))
    assert len(images) == size
    row = [num_inference_steps, size, elapsed, time_per_image]
    df.loc[i] = row
    print(df)
    df.to_csv(outputs, index=False)

