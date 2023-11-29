import numpy as np
np.set_printoptions(linewidth=200, precision=3)

from tinygrad import Device, Tensor
from tinygrad.codegen.linearizer import Linearizer
from tinygrad.graph import print_tree
from tinygrad.helpers import dtypes
from tinygrad.ops import BufferOps, ConstBuffer, LazyOp, LoadOps, MemBuffer, ScheduleItem, UnaryOps
from tinygrad.realize import run_schedule
from tinygrad.shape.shapetracker import ShapeTracker
from tinygrad.shape.symbolic import Variable

device = Device[Device.DEFAULT]

# N = 5
# a = Tensor.rand(10, N).pad((None, (0, 10-N))).contiguous().realize()
# b = (Tensor.rand(10, 10)+1).realize()
# output_st = a.shrink((None, (5, 6))).lazydata.st.unbind()
# for si in (a+b).lazydata.schedule():
#   print(f"{si.ast=}")
# ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.NOOP, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=output_st)),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=output_st))
# prg = device.get_runner(ast)
# prg([a.lazydata.realized, b.lazydata.realized], var_vals={})
# print(a.numpy())

pos = Variable("pos", 1, 10).bind(5)
unbound_pos, val = pos.unbind()

a = Tensor.rand(10, val).pad((None, (0, 10-val))).contiguous().realize()
b = Tensor.arange(100).reshape(10, 10).realize()

output_st = a.shrink((None, (pos, pos+1))).lazydata.st.unbind()
print(f"{output_st=}")

ast = LazyOp(op=BufferOps.STORE, src=(LazyOp(op=UnaryOps.NOOP, src=(LazyOp(op=BufferOps.LOAD, src=(), arg=MemBuffer(idx=1, dtype=dtypes.float, st=output_st)),), arg=None),), arg=MemBuffer(idx=0, dtype=dtypes.float, st=output_st))

# print_tree(ast)

prg = device.get_runner(ast)
# update one column per call
prg([a.lazydata.realized, b.lazydata.realized], var_vals={unbound_pos: 5})
prg([a.lazydata.realized, b.lazydata.realized], var_vals={unbound_pos: 6})
prg([a.lazydata.realized, b.lazydata.realized], var_vals={unbound_pos: 7})

print(a.numpy())

# # does not work yet
# si = ScheduleItem(ast, a.lazydata, (b.lazydata,), {})
# run_schedule([si])
