import os
import sys
import glob
from functools import partial
import time
import threading

import tkinter as tk
from tkinter import (
    Menu,
    Frame,
    Button,
    BOTH,
    LEFT,
    X,
    Label,
    Entry,
    simpledialog,
    ttk,
    filedialog,
    Checkbutton
)
from tkinter.scrolledtext import ScrolledText
from huggingface_hub import login
import sounddevice as sd
import soundfile as sf
import numpy as np
import webrtcvad
import librosa
import whisper
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from models import (
    AVAILABLE_LOCAL_MODELS,
    AVAILABLE_REMOTE_MODELS,
    AVAILABLE_SPEECH_MODELS,
    AVAILABLE_VAD_MODELS,
    load_local_model,
    load_remote_model,
    load_speech_model,
    ModelInterface,
    SpeechModelInterface,
    SAMPLE_RATE,
    CHANNELS,
    AUDIO_DIR
)

if os.path.exists('hf_secret.txt'):
    HF_API_KEY = open('hf_secret.txt', 'r', encoding='utf-8').read().strip()
else:
    HF_API_KEY = None

current_speak_stop_event = None
log_path = 'pipeline_latency.log'

def set_vad_sensitivity(sensitivity):
    global app

    app.vad.set_mode(sensitivity)

    print(f'SYSTEM: VAD sensitivity set to {sensitivity}.\n\n> ', end='')

def clear_chat_history():
    global app

    app.mi.clear_history()

    print(f'SYSTEM: New chat initiated.\n\n> ', end='')

def send_message(dummy=None):
    global app

    msg = app.glob_path_entry.get().strip()

    response = None

    if app.mi.model_name is None:
        print(f'SYSTEM: No model selected yet. Choose from "Local models" or "Remote models" above.\n\n> ', end='')
        return
    else:
        app.glob_path_entry.delete(0, tk.END)
        print(f'YOU: {msg}\n\n> ', end='')
        response = app.mi.call(msg)

    if response:
        print(f'RESPONSE: {response}\n\n> ', end='')
        return response
    else:
        print(f'SYSTEM: Model did not respond.\n\n> ', end='')
        return ''

def stop():
    global current_speak_stop_event
    if current_speak_stop_event:
        current_speak_stop_event.set()

def speak():

    global app, current_speak_stop_event

    if app.smi.model is None:
        print(f'SYSTEM: No speech model selected yet. Choose from "Speech models" above.\n\n> ', end='')
        return

    current_speak_stop_event = threading.Event()

    app.speak_button.config(text="Stop", fg="red", command=stop)

    app.glob_path_entry.delete(0, tk.END)

    def recording_task():
        global app, current_speak_stop_event, log_path

        try:
            audio_data = record_speech(current_speak_stop_event)

            start = time.time()

            loc = save_audio_file(audio_data, 'temp.wav')

            transcription = app.smi.call(loc)
            text = transcription['text'].strip()
            lang = transcription['language']

            if app.state_lang:
                print(f'SYSTEM: Language identified: {lang}.\n\n> ', end='')

            app.glob_path_entry.insert(0, text)

            if app.auto_send:
                response = send_message()
                lat = time.time() - start
                al = transcription['audio_length']
                tc = sum(
                    len(transcription['segments'][i]['tokens'])
                    for i in range(len(transcription['segments']))
                )
                pl = len(app.mi.model.tokenizer.tokenize(
                    app.mi.model.tokenizer.apply_chat_template(
                        app.mi.message_history[:-1],
                        tokenize=False
                    )
                ))
                rl = len(app.mi.model.tokenizer.tokenize(
                    response
                ))
                if app.announce_latency:
                    print(f'SYSTEM: Took {lat:.2} s.\n\n> ', end='')
                if not os.path.exists(log_path):
                    with open(log_path, 'w+', encoding='utf-8') as f:
                        f.write('latency_s\taudio_length_s\tasr_token_count\tllm_input_token_count\tllm_output_token_count\n')
                with open('pipeline_latency.log', 'a+', encoding='utf-8') as f:
                    f.write(f'{lat}\t{al}\t{tc}\t{pl}\t{rl}\n')

        finally:
            app.speak_button.config(text="Speak", fg="green", command=speak)
            current_speak_stop_event = None

    threading.Thread(target=recording_task, daemon=True).start()

def record_speech(stop_event): # Significant help from Claude

    global app

    audio = []

    last_speech_time = time.time()
    frame_size = int(SAMPLE_RATE * 0.03)  # 30ms frames

    def callback(indata, frames, time_info, status):
        nonlocal last_speech_time
        if not stop_event.is_set():
            audio.append(indata.copy())
            
            # Check for speech
            audio_int16 = (indata.flatten() * 32767).astype(np.int16)
            if len(audio_int16) >= frame_size:
                frame_bytes = audio_int16[:frame_size].tobytes()
                if app.vad.is_speech(frame_bytes, SAMPLE_RATE):
                    last_speech_time = time.time()

    with sd.InputStream(
        callback=callback,
        channels=CHANNELS,
        samplerate=SAMPLE_RATE,
        blocksize=frame_size
    ):
        while not stop_event.is_set():
            # Auto-stop after 1.0 second of silence
            if time.time() - last_speech_time > 1.0:
                break
            sd.sleep(100)

    if audio:
        return np.concatenate(audio, axis=0).flatten()
    else:
        return np.array([])
    
def save_audio_file(audio, name=None):

    global app

    os.makedirs(AUDIO_DIR, exist_ok=True)

    if name is None:
        name = 'audio_1'
    
    if not name.endswith('.wav'):
        name_wav += '.wav'
    else:
        name_wav = name
    
    loc = os.path.join(AUDIO_DIR, name_wav)

    if os.path.exists(loc):
        if not loc.startswith('temp') and not loc.endswith('.wav'):
            print(f'SYSTEM: Warning: {loc} already exists.\n\n> ', end='')
        matches = glob.glob(f'{AUDIO_DIR}/{name}_*.wav')
        if matches:
            k = max([
                int(m.split('_')[1][:-4]) for m in matches
            ]) + 1
        else:
            k = 1
        loc = os.path.join(AUDIO_DIR, f'{name}_{k}.wav')
    
    sf.write(loc, audio, SAMPLE_RATE)
    if not loc.startswith('temp'):
        print(f'SYSTEM: Saved audio to {loc}.\n\n> ', end='')

    app.curr_audio = loc
    # print(f'SYSTEM: Selected {app.curr_audio}.\n\n> ', end='')

    return loc

def load_audio_file():
    global app
    app.curr_audio = filedialog.askopenfilename()
    if app.curr_audio is not None:
        print(f'SYSTEM: Selected {app.curr_audio}.\n\n> ', end='')
    else:
        print(f'SYSTEM: Selection canceled.\n\n> ', end='')

def play_audio_file():
    global app

    if app.curr_audio is None:
        print(f'SYSTEM: No audio selected yet. Choose an audio file or speak.\n\n> ', end='')
        return

    data = whisper.load_audio(app.curr_audio, sr=SAMPLE_RATE)

    app.playback_button.config(text="Stop", fg="red", command=sd.stop)

    def playback_task():
        global app
        try:
            sd.play(data, samplerate=SAMPLE_RATE)
            sd.wait()
        finally:
            app.playback_button.config(text="Play current audio", fg="green", command=play_audio_file)

    threading.Thread(target=playback_task, daemon=True).start()

def spectrogram(): # Significant help from Claude
    global app

    if app.curr_audio is None:
        print(f'SYSTEM: No audio selected yet. Load an audio file or speak.\n\n> ', end='')
        return
        
    window = tk.Toplevel()
    window.wm_title('Spectrogram')
    audio = whisper.load_audio(app.curr_audio, sr=SAMPLE_RATE)
    mels = librosa.feature.melspectrogram(
        y=audio,
        sr=SAMPLE_RATE,
        n_fft=app.n_fft,
        hop_length=app.hop_length,
        n_mels=app.n_mels
    )
    mels_db = librosa.power_to_db(mels, ref=np.max)

    fig, ax = plt.subplots(figsize=(10, 4))
    duration = len(audio) / SAMPLE_RATE
    num_frames = mels_db.shape[1]
    time_ticks = np.linspace(0, num_frames, num=6)  # 6 tick marks
    time_labels = [f'{duration * (tick/num_frames):.1f}' for tick in time_ticks]
    sns.heatmap(mels_db, ax=ax, cmap='viridis', cbar_kws={'label': 'Power (dB)'})

    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Mel Frequency Bins')
    ax.set_title('Mel Spectrogram')
    ax.set_xticks(time_ticks)
    ax.set_xticklabels(time_labels)

    fig.tight_layout()

    canvas = FigureCanvasTkAgg(fig, master=window)
    canvas.draw()
    canvas.get_tk_widget().grid(row=0, column=0)

    close_button = ttk.Button(window, text='Close', command=window.destroy)
    close_button.grid(row=1, column=0)

def spectrogram_settings(): # Generated by Claude
    global app
    
    def validate_and_save():
        try:
            n_fft_val = int(n_fft_entry.get())
            hop_length_val = int(hop_length_entry.get())
            n_mels_val = int(n_mels_entry.get())
            
            if n_fft_val <= 0 or hop_length_val <= 0 or n_mels_val <= 0:
                raise ValueError("Values must be positive")
            
            # Update the app settings
            app.n_fft = n_fft_val
            app.hop_length = hop_length_val
            app.n_mels = n_mels_val
            settings_window.destroy()
            
        except ValueError:
            # Clear the error label if it exists and show new error
            error_label.config(text="Invalid input: Both values must be positive integers")
    
    # Create dialog window
    settings_window = tk.Toplevel()
    settings_window.wm_title('Spectrogram Settings')
    
    # n_fft label and entry
    ttk.Label(settings_window, text="n_fft:").grid(row=0, column=1, padx=5, pady=5, sticky='e')
    n_fft_entry = ttk.Entry(settings_window)
    n_fft_entry.insert(0, str(app.n_fft))
    n_fft_entry.grid(row=0, column=1, padx=5, pady=5)
    
    # hop_length label and entry
    ttk.Label(settings_window, text="hop_length:").grid(row=1, column=1, padx=5, pady=5, sticky='e')
    hop_length_entry = ttk.Entry(settings_window)
    hop_length_entry.insert(0, str(app.hop_length))
    hop_length_entry.grid(row=1, column=1, padx=5, pady=5)

    # n_mels label and entry
    ttk.Label(settings_window, text="n_mels:").grid(row=2, column=1, padx=5, pady=5, sticky='e')
    n_mels_entry = ttk.Entry(settings_window)
    n_mels_entry.insert(0, str(app.n_mels))
    n_mels_entry.grid(row=1, column=1, padx=5, pady=5)
    
    # Error label (initially empty)
    error_label = ttk.Label(settings_window, text="", foreground="red")
    error_label.grid(row=2, column=0, columnspan=2, padx=5, pady=5)
    
    # Save button
    save_button = ttk.Button(settings_window, text="Save", command=validate_and_save)
    save_button.grid(row=3, column=0, columnspan=2, pady=10)

def toggle_auto_send():
    global app
    app.auto_send = not app.auto_send
    print(f'SYSTEM: Auto-send set to {app.auto_send}.\n\n> ', end='')

def toggle_announce_latency():
    global app
    app.announce_latency = not app.announce_latency
    print(f'SYSTEM: Announce latency: {app.announce_latency}.\n\n> ', end='')

def toggle_state_lang():
    global app
    app.state_lang = not app.state_lang
    print(f'SYSTEM: State language: {app.state_lang}.\n\n> ', end='')

class PrintLogger(object):  # create file like object

    def __init__(self, textbox: ScrolledText):  # pass reference to text widget
        self.textbox = textbox  # keep ref
        self.textbox.configure(state="disabled")

    def write(self, text):
        self.textbox.configure(state="normal")  # make field editable

        if '\r' in text:
            # Handle carriage return - delete current line and replace
            self.textbox.delete("end-1c linestart", "end-1c")
            # Remove the \r from the text before inserting
            text = text.replace('\r', '')

        self.textbox.insert("end", text)  # write text to textbox
        if '%' in text:
            self.textbox.insert("end", '\n')
        self.textbox.see("end")  # scroll to end
        self.textbox.configure(state="disabled")  # make field readonly
        self.textbox.update()
        
    def clear(self):
        self.textbox.configure(state='normal')
        self.textbox.delete('1.0', tk.END)
        self.textbox.configure(state='disabled')
        print('> ', end='')

    def flush(self):  # needed for file like object
        pass

class MainGUI(tk.Tk):

    def __init__(self):
        tk.Tk.__init__(self)

        self.version = '2.0'

        self.title(f"Chatbot UI v{self.version}")
        w = 896 # width for the Tk root
        h = 504 # height for the Tk root
        # get screen width and height
        ws = self.winfo_screenwidth() # width of the screen
        hs = self.winfo_screenheight() # height of the screen
        # calculate x and y coordinates for the Tk root window
        x = (ws/2) - (w/2)
        y = (hs/2) - (h/2) - 40
        # set the dimensions of the screen
        # and where it is placed
        self.geometry('%dx%d+%d+%d' % (w, h, x, y))
        self.resizable(True, True)
        
        # menu
        self.menu = Menu(self)

        self.mi = ModelInterface()
        self.smi = SpeechModelInterface()
        self.vad = webrtcvad.Vad(1)
        self.curr_audio = None
        self.auto_send = False
        self.announce_latency = False
        self.state_lang = False

        # defaults for spectrogram from librosa
        self.n_fft = 1024
        self.hop_length = 32
        self.n_mels = 128
        
        self.local_menu = Menu(self.menu, tearoff=False)
        for model_name in AVAILABLE_LOCAL_MODELS:
            self.local_menu.add_command(
                label=model_name,
                command=partial(load_local_model, model_name, self.mi)
            )
        self.remote_menu = Menu(self.menu, tearoff=False)
        for model_name in AVAILABLE_REMOTE_MODELS:
            self.remote_menu.add_command(
                label=model_name,
                command=partial(load_remote_model, model_name, self.mi)
            )
        self.speech_menu = Menu(self.menu, tearoff=False)
        for model_name in AVAILABLE_SPEECH_MODELS:
            self.speech_menu.add_command(
                label=model_name,
                command=partial(load_speech_model, model_name, self.smi)
            )
        self.vad_menu = Menu(self.menu, tearoff=False)
        for sensitivity in AVAILABLE_VAD_MODELS:
            self.vad_menu.add_command(
                label=f'Sensitivity {sensitivity}',
                command=partial(set_vad_sensitivity, sensitivity)
            )
        self.audio_menu = Menu(self.menu, tearoff=False)
        self.audio_menu.add_command(
            label='Load audio file',
            command=load_audio_file
        )
        
        self.menu.add_cascade(label='Local models', menu=self.local_menu)
        self.menu.add_cascade(label='Remote models', menu=self.remote_menu)
        self.menu.add_cascade(label='Speech models', menu=self.speech_menu)
        self.menu.add_cascade(label='VAD sensitivity', menu=self.vad_menu)
        self.config(menu=self.menu)
        
        # frames
        self.root = Frame(self)
        self.root.pack(padx=4, pady=4)
        
        # log console
        self.log_console_frame = Frame(self)
        self.log_widget = ScrolledText(
            self.log_console_frame,
            height=4,
            width=20,
            font=("consolas", "16", "normal")
        )
        self.logger = PrintLogger(self.log_widget)
        sys.stdout = self.logger
        sys.stderr = self.logger

        self.button_frame = Frame(self.log_console_frame)
        self.clear_log_button = Button(
            self.button_frame,
            text = "Clear output",
            fg = "black",
            font=("Arial", 16),
            command = self.logger.clear
        )
        self.clear_history_button = Button(
            self.button_frame,
            text = "Clear chat history",
            fg = "black",
            font=("Arial", 16),
            command = clear_chat_history
        )
        self.eval_button = Button(
            self.button_frame,
            text = "Evaluate model",
            fg = "red",
            font=("Arial", 16),
            command = self.mi.evaluate
        )
        self.log_console_frame.pack(fill=BOTH, expand=True)
        self.log_widget.pack(fill=BOTH, expand=True)
        self.clear_log_button.pack(side=LEFT, padx=4)
        self.clear_history_button.pack(side=LEFT, padx=4)
        self.eval_button.pack(side=LEFT, padx=4)
        self.button_frame.pack(pady=4)
        
        # chat message entry
        self.file_path_entry_frame = Frame(self)
        self.glob_path_desc = Label(
            self.file_path_entry_frame,
            text = "Enter chat message:",
            font=("Arial", 16)
        )
        self.glob_path_entry = Entry(
            self.file_path_entry_frame,
            width=20,
            font=("Arial", 16)
        )
        self.send_button = Button(
            self.file_path_entry_frame,
            text = "Send",
            fg = "green",
            font=("Arial", 16),
            command = send_message
        )
        self.speak_button = Button(
            self.file_path_entry_frame,
            text = "Speak",
            fg = "green",
            font=("Arial", 16),
            command = speak
        )
        self.file_path_entry_frame.pack(pady=4, fill=X, expand=False)
        self.glob_path_desc.pack(padx=4, side=LEFT)
        self.glob_path_entry.pack(padx=4, side=LEFT, fill=X, expand=True)
        self.send_button.pack(side=LEFT, padx=4)
        self.speak_button.pack(side=LEFT, padx=4)
        self.bind('<Return>', send_message)

        # audio stuff
        self.playback_frame = Frame(self.log_console_frame)
        self.playback_frame.pack(pady=4, fill=X, expand=False)
        self.audio_buttons = Frame(self.playback_frame)
        self.audio_buttons.pack()
        self.load_audio_button = Button(
            self.audio_buttons,
            text = "Load audio",
            fg = "green",
            font=("Arial", 16),
            command = load_audio_file
        )
        self.playback_button = Button(
            self.audio_buttons,
            text = "Play current audio",
            fg = "green",
            font=("Arial", 16),
            command = play_audio_file
        )
        self.spectrogram_button = Button(
            self.audio_buttons,
            text = "Spectrogram",
            fg = "black",
            font=("Arial", 16),
            command = spectrogram
        )
        self.spectrogram_settings_button = Button(
            self.audio_buttons,
            text = "Spectrogram settings",
            fg = "black",
            font=("Arial", 16),
            command = spectrogram_settings
        )
        self.auto_send_checkbox = Checkbutton(
            self.file_path_entry_frame,
            text = "Auto-send",
            command = toggle_auto_send
        )
        self.announce_latency_checkbox = Checkbutton(
            self.audio_buttons,
            text = "Announce latency",
            command = toggle_announce_latency
        )
        self.state_lang_checkbox = Checkbutton(
            self.button_frame,
            text = "State language",
            command = toggle_state_lang
        )
        
        self.load_audio_button.pack(padx=4, side=LEFT)
        self.playback_button.pack(padx=4, side=LEFT)
        self.spectrogram_button.pack(padx=4, side=LEFT)
        self.spectrogram_settings_button.pack(padx=4, side=LEFT)
        self.announce_latency_checkbox.pack(padx=4, side=LEFT)

        self.auto_send_checkbox.pack(padx=4, side=LEFT)

        self.state_lang_checkbox.pack(padx=4, side=LEFT)

        global HF_API_KEY

        if HF_API_KEY is None:
            HF_API_KEY = simpledialog.askstring(
                title="HuggingFace API Token",
                prompt="Enter your HF API token below:"
            ).strip()
            with open('hf_secret.txt', 'w+', encoding='utf-8') as f:
                f.write(HF_API_KEY)

        login(HF_API_KEY)

        if self.mi.CEREBRAS_API_KEY is None:
            CEREBRAS_API_KEY = simpledialog.askstring(
                title="Cerebras API Token",
                prompt="Enter your Cerebras API token below:"
            ).strip()
            with open('cerebras_secret.txt', 'w+', encoding='utf-8') as f:
                f.write(CEREBRAS_API_KEY)
            self.mi.CEREBRAS_API_KEY = CEREBRAS_API_KEY

        
        print('> ', end='')

def main():
    # breakpoint()
    global app
    app = MainGUI()
    app.mainloop()


if __name__ == "__main__":
    main()