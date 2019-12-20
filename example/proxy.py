class IPM():
    def f(self):
        print("IPM.f()")
    def g(self):
        print("IPM.g()")

class Proxy():
    def __init__(self, handler_cls):
        self.__ipm = handler_cls()
    def __getattr__(self, name):
        return getattr(self.__ipm, name)

p = Proxy(IPM)
p.f()
p.g()
