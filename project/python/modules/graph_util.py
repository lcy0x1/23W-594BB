import numpy.random as npr

"""
Author: Arthur Wang
Date: Mar 5
"""


def rand_list(n, rand=npr.RandomState()):
    ans = [i for i in range(n)]
    for i in range(n):
        a = rand.randint(0, n)
        temp = ans[i]
        ans[i] = ans[a]
        ans[a] = temp
    return ans


def select(n, r, rand=npr.RandomState()):
    ans = [0] * n
    rarr = rand_list(n, rand)
    for i in range(r):
        ans[rarr[i]] = 1
    return ans


def connectivity(n, matrix):
    visited = [0] * n
    queue = []
    
