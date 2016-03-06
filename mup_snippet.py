from multiprocessing import Pool, cpu_count
from multiprocessing.pool import ApplyResult

# --------- see Stenven's solution above -------------
from copy_reg import pickle
from types import MethodType


def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    return _unpickle_method, (func_name, obj, cls)


def _unpickle_method(func_name, obj, cls):
    for cls in cls.mro():
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)


class Myclass(object):

    def __init__(self, nobj, workers=cpu_count()):

        print "Constructor ..."
        # multi-processing
        pool = Pool(processes=workers)
        async_results = [
            pool.apply_async(self.process_obj, (i,)) for i in range(nobj)]
        pool.close()
        # waiting for all results
        map(ApplyResult.wait, async_results)
        lst_results = [r.get() for r in async_results]
        print lst_results

    def __del__(self):
        print "... Destructor"

    def process_obj(self, index):
        print "object %d" % index
        return "results"

pickle(MethodType, _pickle_method, _unpickle_method)
Myclass(nobj=8, workers=3)
# problem !!! the destructor is called nobj times (instead of once)



from multiprocessing import Pool
from functools import partial

def _pickle_method(method):
    func_name = method.im_func.__name__
    obj = method.im_self
    cls = method.im_class
    if func_name.startswith('__') and not func_name.endswith('__'): #deal with mangled names
        cls_name = cls.__name__.lstrip('_')
        func_name = '_' + cls_name + func_name
    return _unpickle_method, (func_name, obj, cls)

def _unpickle_method(func_name, obj, cls):
    for cls in cls.__mro__:
        try:
            func = cls.__dict__[func_name]
        except KeyError:
            pass
        else:
            break
    return func.__get__(obj, cls)

import copy_reg
import types
copy_reg.pickle(types.MethodType, _pickle_method, _unpickle_method)

class someClass(object):
    def __init__(self):
        pass

    def f(self, x=None):
        #can put something expensive here to verify CPU utilization
        if x is None: return 99
        return x*x

    def go(self):
        pool = Pool()
        print pool.map(self.f, range(10))

if __name__=='__main__':
    sc = someClass()
    sc.go()
    x=[someClass(),someClass(),someClass()]
    p=Pool()
    filled_f=partial(someClass.f,x=9)
    print p.map(filled_f,x)
    print p.map(someClass.f,x)


