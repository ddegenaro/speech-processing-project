import os
from time import time

from gtts import gTTS

os.makedirs(os.path.join('tts_eval', 'gtts'), exist_ok=True)

latencies = []

for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
    
        start = time()
    
        tts = gTTS(text)
        tts.save(os.path.join('tts_eval', 'gtts', f'{i}.wav'))
        
        latencies.append(time() - start)
        
with open(os.path.join('tts_eval', 'gtts', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')