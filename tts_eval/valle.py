import os
import sys
sys.path.append('VALL-E-X')
from time import time

import torch
import soundfile as sf
from utils.generation import SAMPLE_RATE, generate_audio, preload_models

# download and load all models
preload_models()

os.makedirs(os.path.join('tts_eval', 'valle'), exist_ok=True)

latencies = []

with torch.no_grad():

    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        
        start = time()
        
        audio_array = generate_audio(text)

        sf.write(
            os.path.join('tts_eval', 'valle', f'{i}.wav'),
            audio_array,
            samplerate=SAMPLE_RATE
        )
        
        latencies.append(time() - start)
        
with open(os.path.join('tts_eval', 'valle', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')