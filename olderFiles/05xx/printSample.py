import sys
import numpy as np

def puts(text):
    sys.stdout.write(text)
    sys.stdout.flush()

def printSample():
    np.set_printoptions(threshold=np.nan)
    m = np.full((160, 150), "0")
    f = open('coordenadasSample3.txt', 'r')
    for line in f:
        a = [int(s) for s in line.split() if s.isdigit()]
        i = a[0]
        j = a[1]
        if a[3] == 50:
            m[i][j] = "."
        elif a[3] == 128:
            m[i][j] = "*"
        elif a[3] == 181:
            m[i][j] = "+"
    
    for i in range(160):
        for j in range(150):
            puts(m[i][j])
        print()

printSample()
