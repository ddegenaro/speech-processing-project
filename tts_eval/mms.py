import os
from time import time

from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

# only ran on CPU
model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

os.makedirs(os.path.join('tts_eval', 'mms'), exist_ok=True)

latencies = []

with torch.no_grad():
    
    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        
        start = time()
    
        inputs = tokenizer(text, return_tensors="pt")
        output = model(**inputs).waveform
        sf.write(
            os.path.join('tts_eval', 'mms', f'{i}.wav'),
            output.cpu().numpy().transpose(),
            samplerate=model.config.sampling_rate
        )
        
        latencies.append(time() - start)
        
with open(os.path.join('tts_eval', 'mms', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')