import ssl
ssl._create_default_https_context = ssl._create_unverified_context

import os

from piper import download_voices

os.makedirs('piper_voices', exist_ok=True)
download_voices.download_voice('en_US-lessac-medium', download_dir='piper_voices')