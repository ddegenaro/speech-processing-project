import os
import sys
from functools import partial
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
    simpledialog
)
from tkinter.scrolledtext import ScrolledText

from huggingface_hub import login

from models import (
    AVAILABLE_LOCAL_MODELS,
    AVAILABLE_REMOTE_MODELS,
    load_local_model,
    load_remote_model,
    ModelInterface
)

if os.path.exists('secret.txt'):
    API_KEY = open('secret.txt', 'r', encoding='utf-8').read().strip()
else:
    API_KEY = None

def clear_chat_history():
    global app

    app.mi.clear_history()

    print(f'SYSTEM: New chat initiated.\n\n> ', end='')

def send_message(dummy=None):
    global app

    msg = app.glob_path_entry.get().strip()

    app.glob_path_entry.delete(0, tk.END)

    response = None

    print(f'YOU: {msg}\n\n> ', end='')

    if app.mi.model_name is None:
        print(f'SYSTEM: No model selected yet. Choose from the menus above.\n\n> ', end='')
    else:
        response = app.mi.call(msg)

    if response:
        print(f'RESPONSE: {response}\n\n> ', end='')

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
        # window

        self.version = '1.0'

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
        
        self.menu.add_cascade(label='Local models', menu=self.local_menu)
        self.menu.add_cascade(label='Remote models', menu=self.remote_menu)
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
        self.file_path_entry_frame.pack(pady=4, fill=X, expand=False)
        self.glob_path_desc.pack(padx=4, side=LEFT)
        self.glob_path_entry.pack(padx=4, side=LEFT, fill=X, expand=True)
        self.send_button.pack(side=LEFT, padx=4)
        self.bind('<Return>', send_message)

        global API_KEY

        if API_KEY is None:
            API_KEY = simpledialog.askstring(
                title="HuggingFace API Token",
                prompt="Enter your HF API token below:"
            )
            with open('secret.txt', 'w+', encoding='utf-8') as f:
                f.write(API_KEY)

        login(API_KEY)
        self.mi.API_KEY = API_KEY
        
        print('> ', end='')

def main():
    # breakpoint()
    global app
    app = MainGUI()
    app.mainloop()


if __name__ == "__main__":
    main()