import numpy as np

def funca(a):
    return a*'a'

def funcb(b):
    return b*'b'

func = [funca, funcb, [funca, funcb]]
print(func[2])