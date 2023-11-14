from transformers import pipeline
import json
import torch
import joblib

pipe = pipeline("text-generation", model="HuggingFaceH4/zephyr-7b-beta", torch_dtype=torch.bfloat16, device="cuda")

def generate_text(prompt, description):
    pipe(prompt, max_new_tokens=512, do_sample=True, temperature=0)
    return outputs[0]["generated_text"]


def create_dataset(path: str):
    splits = ['vald']
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda',
                                            jit=False)  # Must set jit=False for training
    with open(f'../../experiments/{path}/prompt.txt') as f:
        prompt = f.read()

    for split in splits:
        data = AMASS(datapath='../../data/amass_db/babel_30fps_db.pt', clip_preprocess=clip_preprocess,
                     split=split,
                     num_frames=60)
        for i in range(len(data)):
            data.set_clip_text(i, generate_text(prompt))

        joblib.dump(data, f'../../experiments/{path}/dataset_{split}.pt')


if __name__ == '__main__':
    generate_text('Who are you?', 'dance and run')
