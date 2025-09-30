# Chatbot v1.0

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

You can record speech and transcribe it with Whisper by selecting from the "Speech models" menu, and then clicking "Speak." You can stop speaking at any time and the voice activity detection (VAD) will stop recording (you can also click "Stop"). If you find the VAD to be too aggressive, you can select a lower setting from the "VAD sensitivity" menu (0 is the most permissive, 3 is the most aggressive; default 2).

You can then send the transcribed message as if it had been typed out. Check the "Auto-send" option to automatically send your message when you finish speaking.

You can check the "State language" button to confirm that Whisper is correctly detecting the language you spoke. You can also check the "Announce latency" button to see the total time needed to transcribe your speech and fetch an LLM response to your message.

You can clear all output from the screen by clicking "Clear output" if you find it getting too cluttered.

You can clear the language model's memory (prevent it from conditioning on prior chat messages) by clicking "Clear chat history".

Finally, you can subject the language model to an evaluation by clicking the red "Evaluate model" button.

## Spectrograms and audio I/O

You can view the spectrogram of the currently loaded audio file by clicking "Spectrogram." You can adjust a couple of settings used to produce the spectrogram by clicking "Spectrogram settings." See [Librosa's documentation](https://librosa.org/doc/latest/generated/librosa.feature.melspectrogram.html) to understand what they do.

By default, no audio file is loaded. You may load an audio file by clicking "Load audio." If you click "Speak," the audio recorded is saved to a file and loaded automatically.

You can click "Play current audio" at any time to check what is loaded.

## Customizing the evaluation

You can customize the prompts fed to the language model during evaluation and the number of times each prompt is tested. To change the prompts, edit `eval_data.txt` and type one prompt per line.

By default, each prompt is evaluated 10 times. You can change this by editing the following line in `models.py`:

```python
self.num_iters = 10 # for eval
```

## Reading evaluations

Evaluations are automatically saved to `evals/model/path/eval_k.json` where `k` is used to identify different evaluation runs of the same language model.

Information included:

- Loading time
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

## Data visualization

All data visualizations used in the paper can be generated using the IPython notebook `eval_viz.ipynb`. Figures are automatically saved to the `figs` subfolder.
