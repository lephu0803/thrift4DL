import sys
import os

class TModel():
    def __init__(self):
        pass
    
    def get_env(self):
        raise NotImplementedError()

    def preprocessing(self, input):
        raise NotImplementedError()

    def postprocessing(self, input):
        raise NotImplementedError()
    
    def predict(self, request):
        raise NotImplementedError()

