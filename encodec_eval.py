import os
import json

import librosa
import numpy as np
import soundfile as sf
import torch
from transformers import EncodecModel, AutoProcessor

model = EncodecModel.from_pretrained("facebook/encodec_48khz")
processor = AutoProcessor.from_pretrained("facebook/encodec_48khz")

SR = 48_000

data = []
with torch.no_grad():
    for fname in sorted(os.listdir('audio/encodec-eval/originals')):
        
        FROM = os.path.join('audio', 'encodec-eval', 'originals', fname)

        audio_sample, sr = librosa.load(
            FROM,
            sr=sf.info(FROM).samplerate,
            mono=True
        )
        audio_sample = librosa.resample(
            audio_sample,
            orig_sr=sr,
            target_sr=SR
        )
        audio_sample = np.expand_dims(audio_sample, 1).transpose()
        audio_sample = np.repeat(audio_sample, 2, 0)
        audio_sample.shape

        # pre-process the inputs
        inputs = processor(raw_audio=audio_sample, sampling_rate=processor.sampling_rate, return_tensors="pt")

        for target_bandwith in model.config.target_bandwidths:
            
            basename = os.path.basename(FROM)
            f, ext = os.path.splitext(basename)
            TO = FROM.replace(
                basename,
                f + '_' + str(target_bandwith) + ext
            ).replace(
                'originals',
                'reconstructed'
            )
        
            # explicitly encode then decode the audio inputs
            encoder_outputs = model.encode(
                inputs["input_values"],
                inputs["padding_mask"],
                bandwidth=target_bandwith
            )
            enc_out_path = TO.replace(ext, '.json').replace('reconstructed', 'compressed')
            
            enc_out_to_save = {
                'audio_codes': encoder_outputs['audio_codes'].cpu().numpy().tolist(),
                'audio_scales': [x.cpu().numpy().tolist() for x in encoder_outputs['audio_scales']],
                'last_frame_pad_length': encoder_outputs['last_frame_pad_length']
            }
            
            json.dump(
                enc_out_to_save,
                open(enc_out_path, 'w+', encoding='utf-8')
            )
            
            audio_values = model.decode(
                encoder_outputs.audio_codes,
                encoder_outputs.audio_scales,
                inputs["padding_mask"]
            )[0].detach().numpy()[0].transpose()

            sf.write(TO, audio_values, samplerate=SR)
            
            from_size = os.path.getsize(FROM)
            to_size = os.path.getsize(TO)
            enc_rep_size = os.path.getsize(enc_out_path)
            
            from_duration = len(audio_sample[0]) / SR
            to_duration = len(audio_values[:,0]) / SR
            
            data.append({
                'from': FROM,
                'to': TO,
                'from_size_kb': from_size / 1024,
                'to_size_kb': to_size / 1024,
                'audio_compression_ratio': to_size / from_size,
                'encoder_rep_size': enc_rep_size,
                'encoder_compression_ratio': enc_rep_size / from_size,
                'from_duration': from_duration,
                'to_duration': to_duration,
                'bandwith_kbps': target_bandwith,
                'bitrate_kbps': to_size * 8 / to_duration,
                'num_codebooks': encoder_outputs.audio_codes.shape[2]
            })
            
json.dump(
    data,
    open(os.path.join('encodec_eval', 'data.json'), 'w+', encoding='utf-8'),
    indent=4
)