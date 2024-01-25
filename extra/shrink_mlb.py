from tinygrad import Tensor, Device, GlobalCounters
from tinygrad.helpers import Timing

t = Tensor.arange(16).reshape(4, 4).contiguous().realize().shard([f"{Device.DEFAULT}:0", f"{Device.DEFAULT}:1"], axis=0)

print(f"{t.shape=}")
print(f"{t.lazydata.real=}")

# somehow getitem does not work
# a = t[0:2]

a = t.shrink(((0,2),None))
print(a.numpy())
print((a+1).numpy())
print(a.sum(1).numpy())
s = a.sum(0)
print(f"{s.shape=}")
print(s.numpy())

s1 = a.sum(1, keepdim=True)
print(f"{s1.shape=}")
print(s1.numpy())

s2 = s1.mul(0.25)
print(f"{s2.shape=}")
print(s2.numpy())

m = a.mean(1)
print(f"{m.shape=}")
print(m.numpy())

b = t.shrink(((2,4),None))
print(b.numpy())

# this add is a special combine add
c = a * 2 + b / 3.0
print(c.numpy())
