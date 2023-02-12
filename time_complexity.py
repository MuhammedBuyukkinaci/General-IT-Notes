# O(1)

def o_1(arr):
    print("hello")

def func1(x):
    return x+1
def func2(x):
    return x+200000000

# O(n)
def o_n(arr):
    for i in arr:
        print(i)

# O(n^2)
def o_n_2(arr):
    for i in arr:
        for j in arr:
            print(i,j)

# O(n^3)
def o_n_3(arr):
    for i in arr:
        for j in arr:
            for k in arr:
                print(i,j,k)

# O(log n) recursive

def o_log_n_recursive(n):
    if n < 1:
        return "reached to 1"
    else:
        divided_by_2 = n/2
        return o_log_n_recursive(divided_by_2)

print(o_log_n_recursive(8))

# O(log n) iterative

def o_log_n_iterative(n):
    while n > 1:
        n = n / 2
        print(f"{n=}, {n/2=}")
    print("="*50)
o_log_n_iterative(16)
# n=8.0, n/2=4.0
# n=4.0, n/2=2.0
# n=2.0, n/2=1.0
# n=1.0, n/2=0.5

def o_n_log_n(n):
    m = n
    while n >1:
        n = n/2
        for i in range(m):
            print(n, m)

o_n_log_n(4)
#2.0 4
#2.0 4
#2.0 4
#2.0 4
#1.0 4
#1.0 4
#1.0 4
#1.0 4
