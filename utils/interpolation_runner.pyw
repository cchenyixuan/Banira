import os
import sys
from threading import Thread

case = []
for item in os.listdir(sys.argv[1]):
    if os.path.isdir(sys.argv[1] + '//' + item) and item != "banira_files":
        case.append(item)

num = 0
if len(case) > 3:
    if len(case) % 3 == 0:
        num = len(case) // 3
    else:
        num = len(case) // 3 + 1
    for i in range(len(case)//num):
        group = case[i*num:i*num+num]
        group = str(group)
        t = Thread(target=lambda: os.system(f"python .\\data_fitting\\interpolation.py {sys.argv[1]} {group}"))
        t.start()
    if len(case) % 3 != 0:
        group = case[len(case)//num*num:]
        group = str(group)
        t = Thread(target=lambda: os.system(f"python .\\data_fitting\\interpolation.py {sys.argv[1]} {group}"))
        t.start()
else:
    for item in case:
        group = str([item])
        t = Thread(target=lambda: os.system(f"python .\\data_fitting\\interpolation.py {sys.argv[1]} {group}"))
        t.start()

cases = 0
for item in os.listdir(sys.argv[1]):
    if os.path.isdir(sys.argv[1] + '//' + item) and item != "banira_files":
        cases += 1

while True:
    try:
        with open(f"{sys.argv[1]}/banira_files/finished.txt", "r") as f:
            for step, row in enumerate(f):
                assert step != cases-1
            f.close()
    except FileNotFoundError:
        pass
    except AssertionError:
        with open(f"{sys.argv[1]}/banira_files/log.txt", "a") as log:
            log.write(f"All finished\n")
            log.close()
        with open(f"{sys.argv[1]}/banira_files/finished.txt", "a") as log:
            log.write(f"All finished\n{sys.argv[2]}\n")
            log.close()
        break
