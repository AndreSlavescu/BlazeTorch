import torch

'''
available aten operators that get dispatched. 
Used as guideline for writing fusion passes
'''

for op in dir(torch.ops.aten):
    if not op.startswith('_'):
        print(op)
