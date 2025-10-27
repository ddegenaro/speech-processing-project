import os
import re
import json
from glob import glob
from time import time, sleep
import warnings
import psutil

import pandas as pd
import torch
from transformers import pipeline, TextGenerationPipeline, AutomaticSpeechRecognitionPipeline
from cerebras.cloud.sdk import Cerebras

import urllib.request
import ssl
ssl_context = ssl.create_default_context()
ssl_context.check_hostname = False
ssl_context.verify_mode = ssl.CERT_NONE
opener = urllib.request.build_opener(urllib.request.HTTPSHandler(context=ssl_context))
urllib.request.install_opener(opener)
import whisper
import faster_whisper
from transformers import pipeline
import jiwer

SAMPLE_RATE = 16_000
CHANNELS = 1
AUDIO_DIR = 'audio'

AVAILABLE_LOCAL_MODELS = sorted([
    "Qwen/Qwen2.5-0.5B-Instruct",
    "google/gemma-3-270m-it",
    "meta-llama/Llama-3.2-1B-Instruct"
])

AVAILABLE_REMOTE_MODELS = sorted([
    "qwen-3-32b",
    "llama3.1-8b",
    "gpt-oss-120b"
])

AVAILABLE_SPEECH_MODELS = sorted(
    whisper.available_models()
) + ['faster-' + x for x in sorted(
    faster_whisper.available_models()
)] + sorted(['distil-whisper/distil-' + x for x in [
    'large-v2',
    'medium.en',
    'small.en',
    'large-v3.5'
]])

AVAILABLE_VAD_MODELS = [0, 1, 2, 3]

thinking = re.compile(r'<think>[\s\S]*</think>')
WORD = re.compile(r'[A-z0-9ÁÉÍÑÓÚÜáéíñóúü]+')

gb = 1024 * 1024 * 1024

class ModelInterface:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.message_history = []
        self.max_previous_messages = 50

        if os.path.exists('cerebras_secret.txt'):
            self.CEREBRAS_API_KEY = open('cerebras_secret.txt', 'r', encoding='utf-8').read().strip()
        else:
            self.CEREBRAS_API_KEY = None
        self.device = None
        self.num_iters = 5 # for eval
        self.is_remote = None

    def clear_history(self):
        self.message_history = []

    def set_model(self, choice):
        self.model_name = choice
        
        if choice in AVAILABLE_LOCAL_MODELS:
            pipe = pipeline(
                "text-generation",
                choice,
                device_map = "auto",
                dtype = torch.bfloat16
            )

            pipe.generation_config.cache_implementation = "static"

            self.model = pipe
            self.device = self.model.device.type
            self.is_remote = False
        elif choice in AVAILABLE_REMOTE_MODELS:
            client = Cerebras(
                api_key=self.CEREBRAS_API_KEY
            )
            self.model = client
            self.device = None
            self.is_remote = True

        self.message_history = []

    def call(self, prompt: str, clean: bool = True):
        if self.model_name in AVAILABLE_LOCAL_MODELS:
            return prompt_local_model(prompt, self, clean)
        elif self.model_name in AVAILABLE_REMOTE_MODELS:
            return prompt_remote_model(prompt, self, clean)
        else:
            return

    def evaluate(self):

        print(f'SYSTEM: Running evaluation for {self.model_name}...\n\n> ', end='')
        
        if self.model is None:
            print(f'SYSTEM: No model loaded.\n\n> ', end='')
            return
        
        print(f'SYSTEM: Loading {self.model_name}...\n\n> ', end='')

        load_start = time()
        self.set_model(self.model_name)
        load_time = time() - load_start

        print(f'SYSTEM: Loaded {self.model_name} in {load_time:.2}s.\n\n> ', end='')

        try:
            cpu_mem_usage = sum(
                p.numel() * p.element_size() for p in self.model.model.cpu().parameters()
            )
            cpu_mem_usage /= gb
            print(f'SYSTEM: {self.model_name} occupies {cpu_mem_usage:.2} GB on CPU.\n\n> ', end='')
        except:
            cpu_mem_usage = -1.0

        try:
            if self.device == 'mps':
                self.model.model.to('mps')
                gpu_mem_usage = torch.mps.current_allocated_memory()
            elif self.device == 'cuda':
                self.model.model.to('cuda')
                gpu_mem_usage = torch.cuda.memory.memory_allocated()
            else:
                gpu_mem_usage = -1.0
            if self.device in ('mps', 'cuda'):
                gpu_mem_usage /= gb
                print(f'SYSTEM: {self.model_name} occupies {gpu_mem_usage:.2} GB on GPU {self.device}.\n\n> ', end='')
        except:
            gpu_mem_usage = -1.0

        try:
            size_on_disk = os.path.getsize(
                glob(os.path.join(
                    os.environ['HOME'], '.cache', 'huggingface', 'hub',
                    f'models--{self.model_name.replace('/', '--')}',
                    'snapshots', '*', 'model.safetensors'
                ))[0]
            ) / gb
            print(f'SYSTEM: {self.model_name} occupies {size_on_disk:.2} GB on disk.\n\n> ', end='')
        except:
            size_on_disk = -1.0

        overall = {
            'model_name': self.model_name,
            'load_time': load_time,
            'cpu_mem_usage': cpu_mem_usage,
            'gpu_mem_usage': gpu_mem_usage,
            'size_on_disk': size_on_disk,
            'results': []
        }
        
        prompts = open('eval_data.txt', 'r', encoding='utf-8').read().strip().splitlines()
        l_prompts = len(prompts)
        print(f'SYSTEM: Running {l_prompts} prompt(s) {self.num_iters} time(s) each.', end='')
        for j, prompt in enumerate(prompts):

            for i in range(self.num_iters):
                
                message = f'> SYSTEM: Prompt {j+1}/{l_prompts}, iteration {i+1}/{self.num_iters}...'

                print(f'\r{message:<80}', end='', flush=True)

                self.message_history = []

                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()
                
                start = time()
                response = self.call(prompt, clean=False)
                latency = time() - start

                try:
                    if self.device == 'mps':
                        peak_memory = (torch.mps.current_allocated_memory() - gpu_mem_usage) / gb
                    elif self.device == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated() / gb
                    else:
                        peak_memory = -1.0
                except:
                    peak_memory = -1.0

                try:
                    prompt_len = len(self.model.tokenizer.tokenize(
                        self.model.tokenizer.apply_chat_template(
                            self.message_history[:-1],
                            tokenize=False
                        )
                    ))
                    response_len = len(self.model.tokenizer.tokenize(
                        response
                    ))
                except:
                    prompt_len, response_len = -1.0, -1.0

                overall['results'].append({
                    'prompt': prompt,
                    'prompt_len': prompt_len,
                    'response_len': response_len,
                    'iteration': i+1,
                    'response': response,
                    'latency': latency,
                    'peak_memory': peak_memory
                })

                if self.is_remote:
                    sleep(10)

        print('\n\n> SYSTEM: Done.\n\n> ', end='')

        eval_dir = os.path.join(
            'evals', self.model_name
        )
        os.makedirs(eval_dir, exist_ok=True)
        cur_evals = os.listdir(eval_dir)
        if cur_evals:
            k = max([
                int(cur_eval.split('_')[1][:-5]) for cur_eval in cur_evals
            ]) + 1
        else:
            k = 1
        loc = os.path.join('evals', self.model_name, f'eval_{k}.json')
        with open(loc, 'w+', encoding='utf-8') as f:
            json.dump(overall, f, indent=4)
        print(f'SYSTEM: Wrote results to {loc}.\n\n> ', end='')

        self.message_history = []

        return overall

class SpeechModelInterface:
    def __init__(self):
        self.model = None
        self.model_name = None
        self.device = None
        self.setup = None
        self.is_quantized = False

    def set_model(self, choice: str, use_quantized: bool):
        self.model_name = choice
        
        assert choice in AVAILABLE_SPEECH_MODELS

        if torch.backends.mps.is_available():
            self.device = 'mps'
        elif torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # print('USE QUANTIZED:', use_quantized)
        
        if choice in whisper.available_models():

            try:
                self.model = whisper.load_model(choice, device=self.device, in_memory=True)
            except:
                self.device = 'cpu'
                self.model = whisper.load_model(choice, device=self.device, in_memory=True)

            if use_quantized:
                self.model = self.model.bfloat16()
                self.is_quantized = True
                print(f'SYSTEM: Loaded in BF16.\n\n> ', end='')

            self.setup = 'whisper'
        
        elif choice.startswith('faster-'):

            choice = choice.replace('faster-', '')

            if use_quantized:
                compute_type = 'int8'
                self.is_quantized = True
                print(f'SYSTEM: Loaded in INT8.\n\n> ', end='')
            else:
                compute_type = 'default'

            try:
                self.model = faster_whisper.WhisperModel(
                    choice,
                    device=self.device,
                    compute_type=compute_type
                )
            except:
                self.device = 'cpu'
                self.model = faster_whisper.WhisperModel(
                    choice,
                    device=self.device,
                    compute_type=compute_type
                )

            self.setup = 'faster-whisper'

        else:
            
            self.model = pipeline(
                'automatic-speech-recognition',
                model=choice
            )

            if use_quantized:
                self.model.model.bfloat16()
                self.is_quantized = True
                print(f'SYSEM: Loaded in BF16.\n\n> ', end='')

            self.setup = 'hf'

    def call(self, audio: str) -> dict:

        # print('quantized:', self.is_quantized)

        audio = whisper.load_audio(audio, sr=SAMPLE_RATE)
        
        length = len(audio) / SAMPLE_RATE
        
        if self.setup == 'whisper':

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU; using FP32 instead"
                )
                result = self.model.transcribe(audio)

            result['audio_length'] = length

            return result
        
        elif self.setup == 'faster-whisper':

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU; using FP32 instead"
                )
                segments, info = self.model.transcribe(audio)

            segments = [segment for segment in segments]
            segment_dicts = []
            for segment in segments:
                seg_dict = {}
                for key in dir(segment):
                    if not key.startswith('_'):
                        seg_dict[key] = segment.__getattribute__(key)
                segment_dicts.append(seg_dict)

            return {
                'text': ''.join([segment.text for segment in segments]),
                'language': info.language,
                'audio_length': length,
                'segments': segment_dicts
            }

        else:

            with warnings.catch_warnings():
                warnings.filterwarnings(
                    "ignore",
                    message="FP16 is not supported on CPU; using FP32 instead"
                )
                out = self.model(audio, return_timestamps=True)

            out['segments'] = out['chunks']
            del out['chunks']
            out['language'] = 'en'
            for segment in out['segments']:
                segment['tokens'] = self.model.tokenizer.encode(segment['text'])
            out['audio_length'] = length

            return out
        
    def wer(self, refs: list[str], hyps: list[str]):
        clean_refs = [' '.join(re.findall(WORD, ref)).lower() for ref in refs]
        clean_hyps = [' '.join(re.findall(WORD, hyp)).lower() for hyp in hyps]
        return jiwer.wer(clean_refs, clean_hyps)
        
    def prep_data_for_eval(self, df: pd.DataFrame, root: str):
        
        num_audios = 100
        df['sentence_length'] = df['sentence'].str.len()
        df = df.sort_values(by='sentence_length', ascending=False)
        df = df[:num_audios]
        df = df.reset_index()
        assert len(df) == num_audios
        df['transcription'] = ''
        df['audio_length'] = 0.
        df['transcription_length_tokens'] = 0
        df['latency'] = 0.
        df['peak_memory'] = 0.
        df['rtf'] = 0.
        df['wer'] = 0.
        audios = [
            os.path.join(root, 'clips', x) for x in df['path']
        ]
        for audio in audios:
            assert os.path.exists(audio)
        assert len(audios) == num_audios

        return df
    
    def evaluate(self):

        print(f'SYSTEM: Running evaluation for {self.model_name}...\n\n> ', end='')
        
        if self.model_name is None:
            print(f'SYSTEM: No model loaded.\n\n> ', end='')
            return
        
        print(f'SYSTEM: Loading {self.model_name}...\n\n> ', end='')

        load_start = time()
        self.set_model(self.model_name, self.is_quantized)
        load_time = time() - load_start

        print(f'SYSTEM: Loaded {self.model_name} in {load_time:.2}s.\n\n> ', end='')

        if self.setup == 'hf':
            cpu_mem_usage = sum(
                p.numel() * p.element_size() for p in self.model.model.cpu().parameters()
            )
        elif self.setup == 'faster-whisper':
            del self.model
            process = psutil.Process(os.getpid())
            ram_before = process.memory_info().rss
            self.model = faster_whisper.WhisperModel(
                self.model_name.replace('faster-', ''),
                device='cpu'
            )
            ram_after = process.memory_info().rss
            cpu_mem_usage = ram_after - ram_before
            self.set_model(self.model_name, self.is_quantized)
        else:
            cpu_mem_usage = sum(
                p.numel() * p.element_size() for p in self.model.cpu().parameters()
            )
        cpu_mem_usage /= gb
        print(f'SYSTEM: {self.model_name} occupies {cpu_mem_usage:.2} GB on CPU.\n\n> ', end='')

        if self.device == 'mps':
            if self.setup in ('faster-whisper', 'hf'):
                self.model.model.to('mps')
            else:
                self.model.to('mps')
            gpu_mem_usage = torch.mps.current_allocated_memory()
        elif self.device == 'cuda':
            if self.setup in ('faster-whisper', 'hf'):
                self.model.model.to('cuda')
            else:
                self.model.to('cuda')
            gpu_mem_usage = torch.cuda.memory.memory_allocated()
        else:
            gpu_mem_usage = -1.0
        if self.device in ('mps', 'cuda'):
            gpu_mem_usage /= gb
            print(f'SYSTEM: {self.model_name} occupies {gpu_mem_usage:.2} GB on GPU {self.device}.\n\n> ', end='')

        overall = {
            'model_name': self.model_name + '-quantized' if self.is_quantized else self.model_name,
            'load_time': load_time,
            'cpu_mem_usage': cpu_mem_usage,
            'gpu_mem_usage': gpu_mem_usage,
            'size_on_disk': cpu_mem_usage
        }

        # CV - English
        en_ROOT = os.path.join('audio', 'cv-corpus-19.0-delta-2024-09-13', 'en')
        en_data = pd.read_csv(
            os.path.join(en_ROOT, 'validated.tsv'),
            sep='\t'
        )
        en_data = self.prep_data_for_eval(en_data, en_ROOT)

        # CV - Spanish
        es_ROOT = os.path.join('audio', 'cv-corpus-19.0-delta-2024-09-13', 'es')
        es_data = pd.read_csv(
            os.path.join(es_ROOT, 'other.tsv'),
            sep='\t'
        )
        es_data = self.prep_data_for_eval(es_data, es_ROOT)

        # Custom - Wikipedia Titles (English)
        wiki_ROOT = os.path.join('audio', 'wiki-titles')
        wiki_data = pd.read_csv(
            os.path.join(wiki_ROOT, 'meta.tsv'),
            sep='\t'
        )
        wiki_data = self.prep_data_for_eval(wiki_data, wiki_ROOT)

        dataset_tuples = [
            (en_data, 'en_results', en_ROOT),
            (es_data, 'es_results', es_ROOT),
            (wiki_data, 'wiki_results', wiki_ROOT)
        ]
        
        for i, dataset_tuple in enumerate(dataset_tuples):

            print(f'SYSTEM: Running dataset {i+1}/{len(dataset_tuples)}.\n\n> ', end='')

            data, results_key, ROOT = dataset_tuple
            data_len = len(data)

            print(f'SYSTEM: Running {data_len} audio-text pairs.\n\n', end='')
            for j in range(len(data)):
                    
                message = f'> SYSTEM: Audio {j+1}/{data_len}...'

                print(f'\r{message:<80}', end='', flush=True)

                if self.device == 'cuda':
                    torch.cuda.reset_peak_memory_stats()

                audio = os.path.join(ROOT, 'clips', data.at[j, 'path'])
                
                start = time()
                response = self.call(audio)
                latency = time() - start

                try:
                    if self.device == 'mps':
                        peak_memory = (torch.mps.current_allocated_memory() - gpu_mem_usage) / gb
                    elif self.device == 'cuda':
                        peak_memory = torch.cuda.max_memory_allocated() / gb
                    else:
                        peak_memory = -1.0
                except:
                    peak_memory = -1.0

                data.at[j, 'transcription'] = response['text']
                data.at[j, 'audio_length'] = response['audio_length']
                data.at[j, 'transcription_length_tokens'] = sum(
                    [len(segment['tokens']) for segment in response['segments']]
                )
                data.at[j, 'latency'] = latency
                data.at[j, 'peak_memory'] = peak_memory
                data.at[j, 'rtf'] = latency / response['audio_length']
                data.at[j, 'wer'] = self.wer([data.at[j, 'sentence']], [data.at[j, 'transcription']])

            print('\n\n> SYSTEM: Done.\n\n> ', end='')

            model_dir_name = self.model_name.replace(os.path.sep, '--')
            if self.is_quantized:
                model_dir_name += '-quantized'

            eval_dir = os.path.join(
                'speech_evals', model_dir_name
            )
            os.makedirs(eval_dir, exist_ok=True)
            cur_evals = [x for x in os.listdir(eval_dir) if x.startswith(results_key)]
            if cur_evals:
                k = max([
                    int(cur_eval.split('_')[2][:-5]) for cur_eval in cur_evals
                ]) + 1
            else:
                k = 1
            loc = os.path.join(
                'speech_evals', model_dir_name, f'{results_key}_eval_{k}.tsv'
            )
            data.to_csv(loc, sep='\t')
            print(f'SYSTEM: Wrote results to {loc}.\n\n> ', end='')
            overall[results_key] = loc
            overall[f'{results_key}_wer'] = self.wer(
                data['sentence'].tolist(),
                data['transcription'].tolist()
            )
            overall[f'{results_key}_rtf'] = sum(data['latency']) / sum(data['audio_length'])

        cur_evals = [x for x in os.listdir(eval_dir) if x.endswith('json')]
        if cur_evals:
            k = max([
                int(cur_eval.split('_')[2][:-5]) for cur_eval in cur_evals
            ]) + 1
        else:
            k = 1
        with open(
            os.path.join(
                'speech_evals', model_dir_name, f'eval_{k}.json'
            ), 'w+', encoding='utf-8'
        ) as f:
            json.dump(overall, f, indent=4)
        print(f'SYSTEM: Wrote results to {loc}.\n\n> ', end='')

        return overall

def load_local_model(
    choice: str,
    mi: ModelInterface
) -> TextGenerationPipeline:

    if choice not in AVAILABLE_LOCAL_MODELS:
        print(f'SYSTEM (WARNING): {choice} not tested.\n\n> ', end='')
    
    print(f'SYSTEM: Loading {choice}...\n\n> ', end='')
    
    mi.set_model(choice)

    print(f'SYSTEM: Loaded {choice} successfully.\n\n> ', end='')

def load_remote_model(
    choice: str,
    mi: ModelInterface
) -> TextGenerationPipeline:
    
    if choice not in AVAILABLE_REMOTE_MODELS:
        print(f'SYSTEM (WARNING): {choice} not tested.\n\n>', end='')

    print(f'SYSTEM: Connecting to API for {choice}...\n\n>', end='')

    mi.set_model(choice)

    print(f'SYSTEM: Loaded connection to {choice} successfully.\n\n> ', end='')

def prompt_local_model(
    msg: str,
    mi: ModelInterface,
    clean: bool = True
) -> str:
    
    mi.message_history.append({
        "role": "user",
        "content": msg
    })

    if len(mi.message_history) > mi.max_previous_messages:
        mi.message_history.pop(0)

    response = mi.model(
        mi.message_history,
        return_full_text=False
    )[0]['generated_text']

    mi.message_history.append({
        "role": "assistant",
        "content": response
    })
    
    if clean:
        return response.strip()
    else:
        return response

def prompt_remote_model(
    msg: str,
    mi: ModelInterface,
    clean: str = True
) -> str:

    mi.message_history.append({
        "role": "user",
        "content": msg
    })

    response = mi.model.chat.completions.create(
        model=mi.model_name,
        messages=mi.message_history
    ).choices[0].message.content

    mi.message_history.append({
        "role": "assistant",
        "content": response
    })
    
    if clean:
        return re.sub(thinking, '', response).strip()
    else:
        return response

def load_speech_model(
    choice: str,
    smi: SpeechModelInterface,
    app
):
    
    # print(f'USE QUANTIZED (load_speech_model): {app.use_quantized}\n\n> ', end='')

    if choice not in AVAILABLE_SPEECH_MODELS:
        print(f'SYSTEM (WARNING): {choice} not tested.\n\n> ', end='')
    
    print(f'SYSTEM: Loading {choice}...\n\n> ', end='')
    
    smi.set_model(choice, app.use_quantized)

    print(f'SYSTEM: Loaded {choice} successfully.\n\n> ', end='')

    if choice.endswith('.en'):
        print(f'SYSTEM: Warning: {choice} can only transcribe English.\n\n> ', end='')
        