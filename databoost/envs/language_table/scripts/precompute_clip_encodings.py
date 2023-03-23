from databoost.utils.data import find_h5, read_h5, write_h5
import torch
import clip
import numpy as np
import os
import tqdm


DATA_DIR = "/data/sdass/DataBoostBenchmark/language_table/data/separate_oracle_data/processed/"
SUFFIX = "_clip"


device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
clip_model, _ = clip.load("ViT-B/32", device=device)

# find all episodes
file_paths = find_h5(DATA_DIR)
print(f"Augmenting {len(file_paths)} files.")

os.makedirs(DATA_DIR+SUFFIX, exist_ok=True)

# load, add clip, save
for i, file_path in tqdm.tqdm(enumerate(file_paths)):
    data = read_h5(file_path)

    inst = data['infos']['instruction'][0]
    decoded_instruction = bytes(inst[np.where(inst != 0)].tolist()).decode("utf-8")
    text_tokens = clip.tokenize(decoded_instruction)  # .float()[0]  # [77,]
    text_tokens = clip_model.encode_text(
        text_tokens.to(device)).data.cpu().numpy()  # [512,]
    data['observations'] = np.concatenate((
        data['observations'], np.tile(text_tokens, (data['observations'].shape[0], 1))
    ), axis=-1)

    write_h5(data, os.path.join(DATA_DIR + SUFFIX, f"episode_{i}.h5"))


