import blazetorch
import inspect

'''
supported methods in blazeTorch
'''

methods = [attr for attr in dir(blazetorch) if inspect.isfunction(getattr(blazetorch, attr)) or inspect.ismethod(getattr(blazetorch, attr))]
print(methods)
