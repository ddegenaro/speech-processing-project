import os
from time import time

import wave
from piper import PiperVoice

voice = PiperVoice.load("piper_voices/en_US-lessac-medium.onnx")

os.makedirs(os.path.join('tts_eval', 'piper'), exist_ok=True)

latencies = []

for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
    
    start = time()
    
    with wave.open(os.path.join('tts_eval', 'piper', f'{i}.wav'), "wb") as wav_file:
        voice.synthesize_wav(text, wav_file)
        
    latencies.append(time() - start)
    
with open(os.path.join('tts_eval', 'piper', 'latencies.csv'), 'w+', encoding='utf-8') as f:
    for i, latency in enumerate(latencies):
        f.write(f'{i},{latency}\n')