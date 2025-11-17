import os
from time import time

import torch
from parler_tts import ParlerTTSForConditionalGeneration
from transformers import AutoTokenizer
import soundfile as sf

device = "mps" if torch.backends.mps.is_available() else "cpu"

model = ParlerTTSForConditionalGeneration.from_pretrained("parler-tts/parler-tts-mini-v1").to(device)
tokenizer = AutoTokenizer.from_pretrained("parler-tts/parler-tts-mini-v1")

description = "A female speaker delivers a slightly expressive and animated speech with a moderate speed and pitch. The recording is of very high quality, with the speaker's voice sounding clear and very close up."

os.makedirs(os.path.join('tts_eval', 'parler'), exist_ok=True)

latencies = []

with torch.no_grad():
    
    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        
        start = time()

        input_ids = tokenizer(description, return_tensors="pt").input_ids.to(device)
        prompt_input_ids = tokenizer(text, return_tensors="pt").input_ids.to(device)

        generation = model.generate(input_ids=input_ids, prompt_input_ids=prompt_input_ids)
        audio_arr = generation.cpu().numpy().squeeze()
        sf.write(
            os.path.join('tts_eval', 'parler', f'{i}.wav'),
            audio_arr,
            model.config.sampling_rate
        )
        
        latencies.append(time() - start)
        
with open(os.path.join('tts_eval', 'parler', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')