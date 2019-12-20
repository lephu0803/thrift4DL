"""
# References:
1 . http://lucumr.pocoo.org/2009/7/24/singletons-and-their-problems-in-python/
"""
import os
import sys
from multiprocessing import Process
# import tensorflow as tf
import time


print(sys.modules)
## Share state
#class SharedWorker(Process):
#    def __init__(self):
#        # print("Init")
#        Process.__init__(self)
#
#    def run(self):
#        print(id(tf))
#        time.sleep(1)

#for _ in range(3):
#    w = SharedWorker()
#    w.daemon = True
#    w.start()


# Because of daemon initialization, some processes has been killed before executing, so
# we need to hang up main process in a while
#time.sleep(2)
#conclusion = """As we can see, tf module has been shared among process, which leads to some issues in configuration specialization"""
#print(conclusion)


#del tf

class NonSharedWorker(Process):
    def __init__(self):
        # print("Init")
        Process.__init__(self)
        import tensorflow as tf
        self.tf = tf

    def run(self):
        print(id(self.tf))
        time.sleep(1)

for _ in range(3):
    w = NonSharedWorker()
    w.daemon = True
    w.start()
