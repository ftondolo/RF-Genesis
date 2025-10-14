from diffusers import StableDiffusionPipeline
import torch
import matplotlib.pyplot as plt
import numpy as np
import argparse
import time


def get_args():
    # Create the parser
    parser = argparse.ArgumentParser(description='List the content of a folder')
    parser.add_argument('-m', '--model', type=str, default="darkstorm2150/Protogen_x5.3_Official_Release")
    parser.add_argument('-p','--prompt', type=str, default="a living room with a table, a chair, a TV, a computer, a lamp, a plant, a window, a door")
    parser.add_argument('-n', '--name', type=str, default=f"output_{int(time.time())}")
    parser.add_argument('-s', '--steps', type=int, default=25)
    args = parser.parse_args()
    return args.model, args.prompt, args.name, args.steps

model, prompt, name, steps = get_args()

# Load model
pipe = StableDiffusionPipeline.from_pretrained(model,
    torch_dtype=torch.float16,
    safety_checker=None,
).to("cuda")

pipe.load_lora_weights("Asixa/RFLoRA")

image = pipe(prompt, num_inference_steps=steps).images[0]
plt.imsave(name, image)