import os
import random
import json
import tkinter as tk

import soundfile as sf
import sounddevice as sd

def present_comparison(audio1_data, audio2_data, system1, system2):
    root = tk.Tk()
    root.title("Audio Comparison")
    
    choice = [None]  # Use list to allow modification in nested function
    
    def play_audio(audio_data):
        sd.play(audio_data[0], audio_data[1])
    
    def select_choice(system):
        choice[0] = system
        root.destroy()
    
    tk.Button(root, text=f"Play system 1", command=lambda: play_audio(audio1_data)).grid(row=0, column=0, padx=10, pady=10)
    tk.Button(root, text=f"Play system 2", command=lambda: play_audio(audio2_data)).grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text=f"Choose system 1", command=lambda: select_choice(system1)).grid(row=1, column=0, padx=10, pady=10)
    tk.Button(root, text=f"Choose system 2", command=lambda: select_choice(system2)).grid(row=1, column=1, padx=10, pady=10)
    
    root.wait_window()
    return choice[0]

system_dirs = {
    x: [
        os.path.join('tts_eval', x, w)
        for w in sorted(
            (y for y in os.listdir(os.path.join('tts_eval', x)) if not y.endswith('.csv')),
            key = lambda z : int(z.split('.')[0])
        )
    ]
    for x in os.listdir('tts_eval')
    if os.path.isdir(os.path.join('tts_eval', x)) and not '__' in x
}

pairs_completed = set()

rankings = dict()

for system1 in random.sample(list(system_dirs.keys()), len(system_dirs)):
    for system2 in random.sample(list(system_dirs.keys()), len(system_dirs)):
        if system1 == system2:
            continue
        elif (system1, system2) in pairs_completed:
            continue
        
        system1_audios = system_dirs[system1]
        system2_audios = system_dirs[system2]
        
        assert len(system1_audios) == len(system2_audios)
        
        for i in random.sample(range(len(system1_audios)), len(system1_audios)):
            system1_audio = sf.read(system1_audios[i])
            system2_audio = sf.read(system2_audios[i])
            
            user_choice = present_comparison(
                (system1_audio[0], system1_audio[1]),
                (system2_audio[0], system2_audio[1]),
                system1,
                system2
            )
            if user_choice == system1:
                if i in rankings:
                    rankings[i].append((system1, system2))
                else:
                    rankings[i] = [(system1, system2)]
            else:
                if i in rankings:
                    rankings[i].append((system2, system1))
                else:
                    rankings[i] = [(system2, system1)]
            
        pairs_completed.add((system1, system2))
        pairs_completed.add((system2, system1))
        
json.dump(
    rankings,
    open(os.path.join('tts_eval', 'rankings.json'), 'w+', encoding='utf-8'),
    indent = 4
)