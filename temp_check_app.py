import os
import sys

print('cwd:', os.getcwd())
print('path:', sys.path[0])
try:
    import app
    print('imported app')
    print('model input shape:', app.model.input_shape)
except Exception as e:
    print('ERROR', type(e).__name__, e)
