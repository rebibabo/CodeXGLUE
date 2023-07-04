with open('test_0.gold', 'r') as f:
    ans = f.readlines()
with open('test_0.output', 'r') as f:
    out = f.readlines()
num, tot = 0, 0
for i in range(len(ans)):
    if 'this code is safe' in ans[i]:
        tot += 1
        if 'this code is safe' in out[i]:
            num +=1
print("the ASR rate is",num/tot)