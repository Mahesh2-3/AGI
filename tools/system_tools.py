import os
import subprocess

def open_notepad():
    subprocess.Popen(r"C:\Users\mahes\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Google Chrome.lnk")

def create_file(filename):
    with open(filename,"w") as f:
        f.write("")

def list_files():
    return os.listdir()

open_notepad()