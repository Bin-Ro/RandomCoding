import collections

l = list('abcdab')
print(l)
res = collections.Counter(l)
print(res)

for i in res.items():
    print(i)
