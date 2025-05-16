from judgeval.tracer import Tracer

judgment = Tracer(project_name="errors")

@judgment.observe(span_type="func")
def a(a):
    return b(a)

def b(b):
    try:
        return c(b)
    except Exception as e:
        raise Exception("Error in b")

def c(c):
    return d(c)

def d(d):
    raise Exception("Error in d")

@judgment.observe(span_type="func")
def e(e):
    f(e)

def f(f):
    g(f)

def g(g):
    return "OK"



a(1)
e(1)