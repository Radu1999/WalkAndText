from transformers import TextGenerationPipeline, AutoModelForMaskedLM, AutoTokenizer
from src.datasets.amass import AMASS
import json
import clip
import joblib


# model = AutoModelForMaskedLM.from_pretrained('bert-base-uncased')
# tokenizer = AutoTokenizer.from_pretrained('bert-base-uncased')
# pipe = TextGenerationPipeline(model=model, tokenizer=tokenizer, torch_dtype=torch.bfloat16, device="cpu")
#
# prompt = "Tell me 3 swear words: "
#
# outputs = pipe(prompt, max_new_tokens=256, do_sample=True, temperature=0.7, top_k=50, top_p=0.95)

# clip_model, clip_preprocess = clip.load("ViT-B/32", device='cuda',
#                                             jit=False)  # Must set jit=False for training
# test_dataset = AMASS(datapath='../../data/amass_db/babel_30fps_db.pt', clip_preprocess=clip_preprocess, split='vald', num_frames=60)
# test_dataset.set_clip_text(0, 'bro in plm')
# print(test_dataset.__getitem__(0)['clip_text'])
# joblib.dump(test_dataset, '../../experiments/exp1/dataset.pt')


# print(outputs[0]["generated_text"])

def generate_text(prompt):
    return 'text'


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
    create_dataset('exp1')
