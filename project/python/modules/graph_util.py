import numpy.random as npr


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


