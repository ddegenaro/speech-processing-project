import os

from transformers import VitsModel, AutoTokenizer
import torch
import soundfile as sf

model = VitsModel.from_pretrained("facebook/mms-tts-eng")
tokenizer = AutoTokenizer.from_pretrained("facebook/mms-tts-eng")

os.makedirs(os.path.join('tts_eval', 'mms'), exist_ok=True)

with torch.no_grad():
    
    for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
    
        inputs = tokenizer(text, return_tensors="pt")
        output = model(**inputs).waveform
        sf.write(
            os.path.join('tts_eval', 'mms', f'{i}.wav'),
            output.cpu().numpy().transpose(),
            samplerate=model.config.sampling_rate
        )