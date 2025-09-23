import os
import re
import json
from glob import glob
from time import time, sleep

import torch
from transformers import pipeline, TextGenerationPipeline
from cerebras.cloud.sdk import Cerebras

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

thinking = re.compile(r'<think>[\s\S]*</think>')

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