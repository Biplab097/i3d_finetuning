from IPython import display
import numpy as np
import imageio
from tensorflow_docs.vis import embed

def to_gif(images):
    converted_images = np.clip(images * 255, 0, 255).astype(np.uint8)
    imageio.mimsave('./data/animation1.gif', converted_images, fps=25)
    return embed.embed_file('./data/animation1.gif')