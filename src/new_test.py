import numpy as np
sizes = [3,8,10]
biases = [np.random.randn(y, 1) for y in sizes[1:]]
print(sizes[:-1])
weights = [np.random.randn(y, x) for x, y in zip(sizes[:-1], sizes[1:])]
print(weights)

x = 1
y = 1
print(int(x==y))

test_data = [[[1,2,3],[3]],[[1,2,3],[4]]]
for (x, y) in test_data:
    print(x)
    print(y)