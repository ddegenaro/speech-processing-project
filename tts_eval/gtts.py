import os

from gtts import gTTS

os.makedirs(os.path.join('tts_eval', 'gtts'), exist_ok=True)

for i, text in enumerate(open(
        os.path.join('text', 'tts_prompts.txt'), 'r', encoding='utf-8'
    ).readlines()):
        tts = gTTS(text)
        tts.save(os.path.join('tts_eval', 'gtts', f'{i}.wav'))