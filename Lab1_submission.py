## import modules here 

################# Question 0 #################

def add(a, b): # do not change the heading of the function
    return a + b


################# Question 1 #################

def nsqrt(x): # do not change the heading of the function
    l=0
    r=x
    while True:
        i=(l+r)//2
        if i**2==x or (i**2<x and (i+1)**2>x):
            return i
        elif i**2>x:
            r=(l+r)//2
        else:
            l=(l+r)//2


################# Question 2 #################

'''
x_0: initial guess
EPSILON: stop when abs(x - x_new) < EPSILON
MAX_ITER: maximum number of iterations

NOTE: you must use the default values of the above parameters, do not change them
'''
def find_root(f, fprime, x_0=1.0, EPSILON = 1E-7, MAX_ITER = 1000): # do not change the heading of the function
    x=x_0+1
    iter_num=0
    x_new=x_0
    while abs(x-x_new)>=EPSILON and iter_num<=MAX_ITER:
        x=x_new
        x_new=x-f(x)/fprime(x)
        iter_num+=1
    return x_new


################# Question 3 #################

class Tree(object):
    def __init__(self, name='ROOT', children=None):
        self.name = name
        self.children = []
        if children is not None:
            for child in children:
                self.add_child(child)
    def __repr__(self):
        return self.name
    def add_child(self, node):
        assert isinstance(node, Tree)
        self.children.append(node)

def make_tree(tokens): # do not change the heading of the function
    stack=[]
    children=[]
    for i in tokens:
        if i =="]":
            node=stack.pop()
            while node !="[":
                if isinstance(node,str):
                    children.insert(0,Tree(node))
                else:
                    children.insert(0,node)
                node=stack.pop()    
            root=stack.pop()
            stack.append(Tree(root,children))
            children=[]
        else:
            stack.append(i)  
    return stack.pop()    

def max_depth(root): # do not change the heading of the function
    if len(root.children)==0:
        return 1
    depth=[]
    for child in root.children:
        depth.append(max_depth(child))
    return max(depth)+1
