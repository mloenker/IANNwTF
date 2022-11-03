def gen(num, word):
    for j in range(1, num+1):
        yield word * j
    print(" ")


for n in gen(5,'Meow '):
    print(n)