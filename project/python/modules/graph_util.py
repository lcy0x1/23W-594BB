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


def select(n, r, initial_num = 0, assigned_num = 1, rand=npr.RandomState()):
    ans = [initial_num] * n
    rarr = rand_list(n, rand)
    for i in range(r):
        ans[rarr[i]] = assigned_num
    return ans


def check_availability(n):
    return True


def connectivity(pre, hidden, matrix):
    visited = [0] * hidden
    queue = [0]
    visited[0] = 1
    while len(queue) > 0:
        a = queue.pop(0)
        for i in range(hidden):
            if not visited[i] and matrix[a][pre + i]:
                queue.append(i)
                visited[i] = 1
    if sum(visited) < hidden:
        return False

    visited = [0] * hidden
    queue = [0]
    visited[0] = 1
    while len(queue) > 0:
        a = queue.pop(0)
        for i in range(hidden):
            if not visited[i] and matrix[i][pre + a]:
                queue.append(i)
                visited[i] = 1
    if sum(visited) < hidden:
        return False

    return True
