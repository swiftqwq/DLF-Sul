import os
import sys
import time
import subprocess
import signal

#run DLF-Sul.py
def run_DLF_Sul():
    print("Running DLF-Sul.py")
    #subprocess.call(['python3', 'DLF-Sul.py'])
    os.system('python3 DLF-Sul.py')
    #read test_acc as float from test_acc.txt
    with open('test_acc.txt', 'r') as file:
        test_acc = file.read()
    print("test_acc: ", float(test_acc))
    if float(test_acc) > 0.936:
        print("DLF-Sul.py finished with test_acc > 0.936")
        return True
    else:
        print("DLF-Sul.py finished with test_acc < 0.936")
        return False

if __name__ == "__main__":
    while True:
        if run_DLF_Sul():
            break
        else:
            time.sleep(5)
