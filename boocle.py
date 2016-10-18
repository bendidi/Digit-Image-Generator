import os
from subprocess import check_output as qx

for i in range(0,30):
    qx(['python', 'BancTest.py'])
