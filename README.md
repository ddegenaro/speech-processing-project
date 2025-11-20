# Chatbot v3.0

## Setup

Run the following command:

```bash
git clone https://github.com/ddegenaro/speech-processing-project.git
```

Create a Python environment as you wish, or use one you already like. Ensure `tkinter` is installed like so:

```bash
python -c "import tkinter"
```

If this command exits without displaying a Traceback, `tkinter` is installed.

Navigate into the project directory with

```bash
cd speech-processing-project
```

and install the appropriate packages (or skip this step if you think the versions you have are sufficient):

```bash
python -m pip install -r requirements.txt
```

Additionally, be sure to install `ffmpeg`. On a Mac, this can usually be installed with `brew`:

```bash
brew install ffmpeg
```

On other OS, the process can be a bit more complicated. A great tutorial can be found [on the page for a project I authored](https://ddegenaro.github.io/whisper-ui/).

Finally, run the following to open the user interface:

```bash
python ui.py
```

You will be asked to provide a HuggingFace API Token in order to access gated models. To do this, make an account at [HuggingFace](https://huggingface.co/). Then, head to [HuggingFace settings](https://huggingface.co/settings/tokens) and create a new "Read-only" token. Copy and paste the entire thing in the box shown in the UI. Avoid sharing this token with anyone.

You will also be asked to provide a Cerebras API Token in order to access the Cerebras API (remote models). To do this, make an account at [Cerebras](https://cloud.cerebras.ai/?utm_source=inferencedocs). Then, head to [the Cerebras Cloud Platform](https://cloud.cerebras.ai/platform/), navigate to "API keys" on the left-hand side, and copy and paste your "Default Key" here. Avoid sharing this token with anyone.

## Model selection

Only 6 language models (3 local, 3 using free API inference providers) have been tested. These are accessible from the menu bar at the top of the window/screen (depending on your OS).

## Chatting and speaking

Type a message into the text entry at the bottom of the screen and press enter or click the "Send" button to send it to the language model you've selected.

Responses can be read aloud via Piper TTS by clicking the "Read" button. To automatically read messages aloud, check the "Auto-speak" checkbox.

You can record speech and transcribe it with Whisper by selecting from the "Speech models" menu, and then clicking "Speak." You can stop speaking at any time and the voice activity detection (VAD) will stop recording (you can also click "Stop"). If you find the VAD to be too aggressive, you can select a lower setting from the "VAD sensitivity" menu (0 is the most permissive, 3 is the most aggressive; default 2).

You can then send the transcribed message as if it had been typed out. Check the "Auto-send" option to automatically send your message when you finish speaking. If checked, this option will also record latency information about the overall pipeline to `pipeline_latency.log`.

You can check the "State language" button to confirm that Whisper is correctly detecting the language you spoke. You can also check the "Announce latency" button to see the total time needed to transcribe your speech and fetch an LLM response to your message.

You can clear all output from the screen by clicking "Clear output" if you find it getting too cluttered.

You can clear the language model's memory (prevent it from conditioning on prior chat messages) by clicking "Clear chat history".

You can subject the language model to an evaluation by clicking the red "Evaluate text model" button, or evaluate your currently selected Whisper variant by clicking the red "Evaluate speech model" button.

## Spectrograms and audio I/O

You can view the spectrogram of the currently loaded audio file by clicking "Spectrogram." You can adjust a couple of settings used to produce the spectrogram by clicking "Spectrogram settings." See [Librosa's documentation](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) to understand what they do.

By default, no audio file is loaded. You may load an audio file by clicking "Load audio." If you click "Speak," the audio recorded is saved to a file and loaded automatically.

You can click "Play current audio" at any time to check what is loaded.

## Customizing the language model evaluation

You can customize the prompts fed to the language model during evaluation and the number of times each prompt is tested. To change the prompts, edit `eval_data.txt` and type one prompt per line.

By default, each prompt is evaluated 10 times. You can change this by editing the following line in `models.py`:

```python
self.num_iters = 10 # for eval
```

## Speech model evaluation (relatively fixed)

The speech models are evaluated on three small datasets:

- The 100 longest (by gold transcript character count) validated audio-transcript pairs in the English CommonVoice v19.0 delta segment
- The 100 longest (by gold transcript character count) "other" audio-transcript pairs in the English CommonVoice v19.0 delta segment (too few are validated)
- 100 recordings of the project author reading 100 pseudo-randomly selected English Wikipedia article titles (1 title per recording, titles with learned borrowings and rare/foreign names excluded)

## Reading language model evaluations

Language model evaluations are automatically saved to `evals/model/path/eval_k.json` where `k` is used to identify different evaluation runs of the same language model.

Information included:

- Loading time (s)
- CPU memory usage (GB, -1.0 if not local)
- GPU memory usage (GB, -1.0 if no GPU or not local)
- Size on disk (GB, -1.0 if not local)
- Information about each response:
  - Prompt given
  - Length of prompt in tokens (including chat template, -1.0 if not local)
  - Length of generated response in tokens (not including prompt, -1.0 if not local)
  - Iteration (if executing each prompt several times)
  - Response generated
  - Latency in seconds (subtract 3 for remote due to `time.sleep` calls used to prevent rate limiting)
  - Estimated peak memory usage (GB, -1.0 if no GPU or not local)

## Reading speech model evaluations

Speech model evaluations are automatically saved to `speech_evals/model/eval_k.json` where `k` is used to identify different evaluation runs of the same language model. Sample-wise results are also generated and saved as `speech_evals/model/ds_results_eval_k.tsv`, where `ds` is either `en` for English CommonVoice, `es` for Spanish CommonVoice, or `wiki` for Wikipedia titles.

Information included in the `.json` file:

- Model name
- Load time (s)
- CPU memory usage (GB)
- GPU memory usage (GB, -1.0 if no GPU or if implementation not compatible with system GPU)
- Size on disk (GB)
- Path to `.tsv` containing sample-wise results for English CommonVoice
- Word error rate for English CommonVoice (out of 1.0)
- Path to `.tsv` containing sample-wise results for Spanish CommonVoice
- Word error rate for Spanish CommonVoice (out of 1.0)
- Real-time factor for Spanish CommonVoice (transcription time / real time)
- Path to `.tsv` containing sample-wise results for Wikipedi titles
- Word error rate for Wikipedia titles (out of 1.0)
- Real-time factor for Wikipedia titles (transcription time / real time)

Sample-wise information included in each `.tsv` file:

- File name (`path` - inside `audio/ds/clips` where `ds` is either `cv-corpus-19.0-delta-2024-09-13/en`, `cv-corpus-19.0-delta-2024-09-13/es`, or `wiki-titles`)
- Gold transcript (`sentence`)
- Whisper transcript (`transcription`)
- Audio length (s) (`audio_length`)
- Transcription length in tokens (`transcription_length_tokens`)
- Latency (s) (`latency`)
- Real-time factor (transcription time / real time) (`rtf`)
- Word error rate (`wer`)

Note: in this repository, all data relating to the Wikipedia titles is included due to the dataset being the author's own work. However, CommonVoice is not to be redistributed without the permission of Mozilla. As such, to re-produce any results relating to CommonVoice, it's necessary to pull your own copy of CommonVoice. The `audio` folder must have the following structure should you choose to do this:

```markdown
audio/
- cv-corpus-19.0-delta-2024-09-13/
  - en/
    - clips/
      - common_voice_en_40865211.mp3
      - ...
    clip_durations.tsv
    invalidated.tsv
    other.tsv
    reported.tsv
    unvalidated_sentences.tsv
    validated_sentences.tsv
    validated.tsv
  - es/
    - clips/
      - common_voice_es_40867528.mp3
      - ...
    clip_durations.tsv
    invalidated.tsv
    other.tsv
    reported.tsv
    unvalidated_sentences.tsv
    validated_sentences.tsv
    validated.tsv
- wiki-titles/
  - clips/
    - 1.m4a
    - ...
  - meta.tsv
```

## Data visualization

All data visualizations used in the first paper can be generated using the IPython notebook `eval_viz.ipynb`. Figures are automatically saved to the `figs` subfolder.

All data visualizations used in the second paper are handled similarly and can be generated using the IPython notebooks `spectrogram_viz.ipynb` and `latency_model.ipynb`.

All data visualizations used in the fourth paper are handled similarly and can be generated using the IPython notebooks `encodec_viz.ipynb` and `tts_viz.ipynb`.

## Encodec evaluation

`encodec_eval.py` will run the 48 kHz FAIR Encodec model on all audio files stored at `audio/encodec-eval/originals`. It will create two other subdirectories, `audio/encodec-eval/compressed`, where the audio encoding is written as a JSON with near optimal compression (while still being readable), and `audio/encodec-eval/compressed`, where the encodings are passed through the decoder to reconstruct the original audio as closely as possible. It will also generate `encodec_eval/data.json`, which stores useful statistics about each Encodec run:

- `'from'`: the path to the original audio file.
- `'to'`: the path to the reconstructed audio file under consideration.
- `'from_size_kb'`: the size of the original audio file, in KB.
- `'to_size_kb'`: the size of the reconstructed audio file, in KB.
- `'audio_compression_ratio'`: the ratio of the above two (less than 1 if compression achieved).
- `'encoder_rep_size'`: the size of the JSON file storing the encoding, in bytes (accident - should be KB for comparison).
- `'encoder_compression_ratio'`: the ratio of the size of the JSON file to the original audio (generally much better compression than `audio_compression_ratio`).
- `'from_duration'`: the length of the original audio file, in seconds.
- `'to_duration'`: the length of the reconstructed audio file, in seconds.
- `'bandwith_kbps'`: the bandwidth (typo in the name) used by the model in this instance, corresponding to more or fewer codebooks.
- `'bitrate_kbps'`: the bitrate of the reconstructed audio file, in Kb/s, defined as the size of the file in bits divided by its duration.
- `'num_codebooks'`: effectively an alias of `'bandwith_kbps`, having a 1-to-1 mapping.

## TTS systems

Prompts for TTS Evaluation can be found in `text/tts_prompts.txt`, one prompt per line. 15 prompts are included. When running any of the following TTS systems, they will be given each of those prompts. Generated audio files will be saved to `tts_eval/<name>`, where `<name>` is the name of the system (the same name as each of the Python scripts that follow).

### Bark

Fully implemented in HF. Just run `tts_eval/bark.py`.

### gTTS

Requires installation of `gtts` via `pip`. May raise a circular import error. Simply run all the lines from `tts_eval/gtts.py` in an interactive session instead.

### MMS

Fully implemented in HF. Just run `tts_eval/mms.py`.

### Parler

Requires installation of `parler-tts` via `pip`. Then, run `tts_eval/parler.py`.

### Piper

Requires installation of `piper` via `pip`. Start by running `tts_eval/piper_setup.py`, which will download a voice embedding.

In your installation of `piper`, depending on your Python version, you may need to make these changes:

- Add `import os` to the top of the file
- Change `model_path = download_dir / f"{voice_code}.onnx"` to `model_path = Path(os.path.join(download_dir, f"{voice_code}.onnx"))`
- Change `config_path = download_dir / f"{voice_code}.onnx.json"` to `model_path = Path(os.path.join(download_dir, f"{voice_code}.onnx"))`

Finally, run `tts_eval/piper.py`. This library may also have a circular import error. If so, just run the lines in an interactive session.

### VALL-E-X

Make the VALL-E-X repo a subdirectory of this project's directory. Follow the instructions at [`https://github.com/Plachtaa/VALL-E-X?tab=readme-ov-file#-installation`](https://github.com/Plachtaa/VALL-E-X?tab=readme-ov-file#-installation).

In `VALL-E-X/utils.generation.py`, make the following modifications:

- Change `tokenizer_path="./utils/g2p/bpe_69.json"` to `tokenizer_path="VALL-E-X/utils/g2p/bpe_69.json"`.
- Change `checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu')` to `checkpoint = torch.load(os.path.join(checkpoints_dir, model_checkpoint_name), map_location='cpu', weights_only=False)`.

Finally, run `tts_eval/valle.py`.

## TTS Evaluation

After running all the `tts_eval/*.py` scripts, one may run `a_b_test_tts.py` to get an interactive annotation interface. You may play the speech as many times as you like before selecting one of the systems. Annotations will be stored in `tts_eval/rankings.json`. Finally, one can run `elo.py` to produce `tts_eval/elo_rankings.csv`, which yields an aggregate score for each system.
