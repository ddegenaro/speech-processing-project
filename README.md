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

Run the following to open the user interface:

```bash
python ui.py
```

You will be asked to provide a HuggingFace API Token in order to access gated models. To do this, make an account at [https://huggingface.co/](https://huggingface.co/). Then, head to [https://huggingface.co/settings/tokens](https://huggingface.co/settings/tokens) and create a new "Read-only" token. Copy and paste the entire thing in the box shown in the UI. Avoid sharing this token with anyone.

You will also be asked to provide a Cerebras API Token in order to access the Cerebras API (remote models). To do this, make an account at [https://cloud.cerebras.ai/?utm_source=inferencedocs](https://cloud.cerebras.ai/?utm_source=inferencedocs). Then, head to [https://cloud.cerebras.ai/platform/](https://cloud.cerebras.ai/platform/), navigate to "API keys" on the left-hand side, and copy and paste your "Default Key" here. Avoid sharing this token with anyone.

## Model selection

Only 6 models (3 local, 3 using free API inference providers) have been tested. These are accessible from the menu bar at the top of the window/screen (depending on your OS).

## Chatting

Type a message into the text entry at the bottom of the screen and press enter or click the "Send" button to send it to the model you've selected.

You can clear the output from the screen by clicking "Clear output" if you find it getting too cluttered.

You can clear the model's memory (prevent it from conditioning on prior chat messages) by clicking "Clear chat history".

Finally, you can subject the model to an evaluation by clicking the red "Evaluate model" button.

## Customizing the evaluation

You can customize the prompts fed to the model and the number of times each prompt is tested. To change the prompts, edit `eval_data.txt` and type one prompt per line.

By default, each prompt is evaluated 10 times. You can change this by editing the following line in `models.py`:

```python
self.num_iters = 10 # for eval
```

## Reading evaluations

Evaluations are automatically saved to `evals/model/path/eval_k.json` where `k` is used to identify different evaluation runs of the same model.

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