from transformers import pipeline
import json
import torch
import joblib
from src.datasets.amass import AMASS

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device="cuda")


def generate_text(prompt, inputs):
    prompts = [prompt + input for input in inputs]
    outputs = pipe(prompts, batch_size=8, max_new_tokens=512, do_sample=True, temperature=0.5)
    return [output[0]["generated_text"] for output in outputs]


def create_dataset(path: str):
    splits = ['vald']
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda',
                                            jit=False)  # Must set jit=False for training
    with open(f'../../experiments/{path}/prompt.txt') as f:
        prompt = f.read()

    for split in splits:
        inputs = []
        data = AMASS(datapath='../../data/amass_db/babel_30fps_db.pt', clip_preprocess=clip_preprocess,
                     split=split,
                     num_frames=60)
        for i in range(len(data)):
            if i == 3:
                break
            inputs.append(data._get_item_data_index(i)['clip_text'])

        print(generate_text(prompt, inputs))
        joblib.dump(data, f'../../experiments/{path}/dataset_{split}.pt')


if __name__ == '__main__':
    create_dataset('exp1')
