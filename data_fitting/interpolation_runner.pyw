import os
import sys
from threading import Thread


case_all = sys.argv[2].split(",")[:-1]
case = sys.argv[3].split(",")[:-1]
case_fin = [item for item in case_all if item not in case]

# case_all = sorted(sys.argv[2].split(",")[:-1])
# case = sorted(sys.argv[3].split(",")[:-1])
# case_fin = []
# ptr = 0
# for i in case:
#     while case_all[ptr] != i:
#         case_fin.append(case_all[ptr])
#         ptr += 1
#     ptr += 1
# case_fin.extend(case_all[ptr:])  O(n2) -> O(n)

if case:
    with open(f"{sys.argv[1]}/banira_files/finished.txt", "w") as f:
        for item in case_fin:
            f.write(f"{item}\n")
        f.close()

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

    cases = len(case)

    while True:
        cnt = 0
        try:
            with open(f"{sys.argv[1]}/banira_files/finished.txt", "r") as f:
                for step, row in enumerate(f):
                    if row[:-1] in case:
                        cnt += 1
                f.close()
            assert cnt != cases
        except FileNotFoundError:
            pass
        except AssertionError:
            with open(f"{sys.argv[1]}/banira_files/log.txt", "a") as log:
                log.write(f"All finished\n")
                log.close()
            with open(f"{sys.argv[1]}/banira_files/finished.txt", "a") as log:
                log.write(f"All finished\n")
                log.close()
            break
