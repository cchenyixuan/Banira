import os
import sys
from threading import Thread

case = []

try:
    sys.argv[2]
    case = sys.argv[2].split(",")

    group = str(case)
    t = Thread(target=lambda: os.system(f"python .\\training_process\\preproceed.py {sys.argv[1]} {group}"))
    t.start()

    cases = len(case)

    while True:
        cnt = 0
        try:
            with open(f"{sys.argv[1]}/banira_files/finished_ex.txt", "r") as f:
                for step, row in enumerate(f):
                    if row[:-1] in case:
                        cnt += 1
                f.close()
                assert cnt != cases
        except FileNotFoundError:
            pass
        except AssertionError:
            with open(f"{sys.argv[1]}/banira_files/finished_ex.txt", "a") as f:
                f.write(f"All finished\n")
                f.close()
            break

except IndexError:
    pass
