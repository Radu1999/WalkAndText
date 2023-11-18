from transformers import pipeline
import json
import time
import torch
import joblib
from src.datasets.amass import AMASS
import clip
pipe = pipeline("text-generation", model="mistralai/Mistral-7B-Instruct-v0.1", torch_dtype=torch.bfloat16, device="cuda")
pipe.tokenizer.pad_token_id = pipe.model.config.eos_token_id

def generate_text(prompt, inputs):
    prompts = [prompt + inp + ' [/INST]' for inp in inputs]
    outputs = pipe(prompts, batch_size=8, max_new_tokens=512, do_sample=True, temperature=0.5)
    return [output[0]["generated_text"].split('[/INST]')[-1] for output in outputs]


def create_dataset(path: str):
    splits = ['vald']
    clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda',
                                            jit=False)  # Must set jit=False for training
    with open(f'experiments/{path}/prompt.txt') as f:
        prompt = f.read()

    total_examples = 0
    for split in splits:
        inputs = []
        data = AMASS(datapath='data/amass_db/babel_30fps_db.pt', clip_preprocess=clip_preprocess,
                     split=split,
                     num_frames=60)
        total_exampes += len(data)
        for i in range(len(data)):
            inputs.append(data._get_item_data_index(i)['clip_text'])

        texts = generate_text(prompt, inputs)
        for i in range(len(data)):
            data.set_clip_text(i, texts[i])

        joblib.dump(data, f'experiments/{path}/dataset_{split}.pt')
    return total_examples

if __name__ == '__main__':
    start = time.time()
    count = create_dataset('exp1')
    print(f'Data generated! For {count} examples it took {time.time() - start seconds}')
