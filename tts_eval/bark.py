import os
from time import time

from transformers import pipeline
import torch
import soundfile as sf

pipe = pipeline("text-to-speech", model="suno/bark")

os.makedirs(os.path.join('tts_eval', 'bark'), exist_ok=True)

latencies = []

with torch.no_grad():
    
    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        
        start = time()
    
        output = pipe(text)
        sf.write(
            os.path.join('tts_eval', 'bark', f'{i}.wav'),
            output['audio'].transpose(),
            samplerate=output['sampling_rate']
        )
        
        latencies.append(time() - start)
        
with open(os.path.join('tts_eval', 'bark', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')