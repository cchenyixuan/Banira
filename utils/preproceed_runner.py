import os
import sys
from threading import Thread

path = r'C:\Users\cchen\PycharmProjects\seika_numerical'
for item in os.listdir(path):
    if os.path.isdir(path + '//' + item):
        t = Thread(target=lambda: os.system(f"python preproceed.py {path + '//' + item} {path}"))
        t.start()
