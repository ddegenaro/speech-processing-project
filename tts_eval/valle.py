import os
import sys
sys.path.append('VALL-E-X')

import torch
import soundfile as sf
from utils.generation import SAMPLE_RATE, generate_audio, preload_models

# download and load all models
preload_models()

os.makedirs(os.path.join('tts_eval', 'vall-e-x'), exist_ok=True)

with torch.no_grad():

    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        audio_array = generate_audio(text)

        sf.write(
            os.path.join('tts_eval', 'vall-e-x', f'{i}.wav'),
            audio_array,
            samplerate=SAMPLE_RATE
        )