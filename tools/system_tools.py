import os
import subprocess

def open_notepad():
    subprocess.Popen("notepad.exe")

def create_file(filename):
    with open(filename,"w") as f:
        f.write("")

def list_files():
    return os.listdir()