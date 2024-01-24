from __future__ import annotations
from typing import Optional, Union, Any, Tuple, List
import functools, itertools
from tinygrad.helpers import all_same, dedup, round_up, DEBUG
from tinygrad.dtype import DType
from tinygrad.ops import BinaryOps, LoadOps, UnaryOps, TernaryOps, ReduceOps
from tinygrad.lazy import LazyBuffer, create_schedule
from tinygrad.shape.shapetracker import ShapeTracker, sint

def all_reduce(op:ReduceOps, lbs):
  # TODO: replace this with ring reduce
  bop = {ReduceOps.SUM:BinaryOps.ADD, ReduceOps.MAX:BinaryOps.MAX}[op]
  return [functools.reduce(lambda x,y: x.e(bop, y), [x.copy_to_device(lb.device) for x in lbs]) for lb in lbs]

def to_sharded(lbs:List[LazyBuffer], axis:int) -> List[LazyBuffer]:
  if DEBUG >= 3 and lbs[0].shape[axis] % len(lbs) != 0: print(f"multi axis uneven: {lbs[0].shape=} {axis=} {len(lbs)=}")
  sz = round_up(lbs[0].shape[axis], len(lbs)) // len(lbs)
  return [lb.shrink(tuple((0,s) if a != axis else (sz*i,min(s,sz*(i+1))) for a,s in enumerate(lb.shape))) for i,lb in enumerate(lbs)]

class MultiLazyBuffer:
  def __init__(self, lbs:List[LazyBuffer], axis:Optional[int], real:Optional[List[int]]=None):
    assert all(isinstance(x, LazyBuffer) for x in lbs) and len(lbs), "all lbs must be LazyBuffers, and we need at least one of them"
    #assert all_same([(x.shape, x.dtype, x.st) for x in lbs]), "all multilazybuffer needs same shape, dtype, and st"
    self.lbs, self.axis, self.dtype, self.device, self.real = lbs, axis, lbs[0].dtype, tuple(x.device for x in lbs), real or [True]*len(lbs)
    if axis is not None:
      splits = list(itertools.accumulate([lb.shape[axis] for lb in lbs], initial=0))
      self.bounds = [(st,ed) for st,ed in zip(splits, splits[1:])]

  @property
  def shape(self): return tuple(sum(y.shape[a] for y in self.real_lbs) if a == self.axis else s for a,s in enumerate(self.real_lbs[0].shape))

  @property
  def size(self): return sum(x.size for x in self.real_lbs)

  @property
  def real_lbs(self): return [lb for lb,r in zip(self.lbs, self.real) if r]

  def __repr__(self):
    return f"<MLB {self.axis=}{chr(10)}{chr(10).join([f'{x.device} {x.st}' for x in self.lbs])}>"

  @staticmethod
  def from_sharded(lb:LazyBuffer, devices:Tuple[str, ...], axis:Optional[int]=None):
    lbs = [lb.contiguous() if lb.base != lb else lb] * len(devices)
    return MultiLazyBuffer([lb.copy_to_device(d).contiguous() for lb,d in zip(to_sharded(lbs, axis) if axis is not None else lbs, devices)], axis)

  def copy_to_device(self, device:str) -> LazyBuffer:
    if self.axis is None: return self.lbs[0].copy_to_device(device)
    sz = self.real_lbs[0].shape[self.axis]
    llbs = []
    for i,lb in enumerate([lb.copy_to_device(device) for lb in self.real_lbs]):
      pad_arg = tuple((0,0) if a != self.axis else (sz*i, max(0, self.shape[self.axis]-sz*(i+1))) for a in range(len(lb.shape)))
      llbs.append(lb.pad(pad_arg))
    return functools.reduce(lambda x,y: x.e(BinaryOps.ADD, y), llbs)

  # TODO: fix this
  def is_unrealized_contiguous_const(self): return False

  # passthroughs
  def schedule(self, seen=None): return create_schedule(self.real_lbs, seen)
  def cast(self, dtype:DType, bitcast:bool=False): return MultiLazyBuffer([x.cast(dtype, bitcast) for x in self.lbs], self.axis, self.real)
  def const(self, val:Union[float, int], dtype:Optional[DType]=None) -> MultiLazyBuffer:
    return MultiLazyBuffer([x.const(val, dtype=dtype) for x in self.lbs], self.axis, self.real)
  def contiguous(self): return MultiLazyBuffer([x.contiguous() for x in self.lbs], self.axis, self.real)

  # elementwise is simple
  def e(self, op:Union[LoadOps, UnaryOps, BinaryOps, TernaryOps], *in_srcs:MultiLazyBuffer, arg:Optional[Any]=None) -> MultiLazyBuffer:
    msrcs = (self,)+in_srcs
    assert all(isinstance(x, MultiLazyBuffer) for x in msrcs), f"all buffers must be MultiLazyBuffer {msrcs}"
    print(f"\n e {op=}")
    # print(f"{msrcs=}")
    # print(f"{self.real=}")

    # print(f"{self.real=}")
    # print(f"{self.device=}")
    # if in_srcs:
    #   print(f"{in_srcs[0].device=}")
    #   print(f"{in_srcs[0].real=}")

    if any(not all(src.real) for src in msrcs):
      out_real = [False] * len(self.real)
      out_lbs = [[] for _ in self.real]
      # print(f"{msrcs=}")
      for src in msrcs:
        # print(f"{src.real=}")
        for i, (lb, r) in enumerate(zip(src.lbs, src.real)):
          if r:
            out_real[i] = True
            out_lbs[i].append(lb)
      for i, (lbs,r) in enumerate(zip(out_lbs,out_real)):
        out_lbs[i] = self.lbs[i] if not lbs or not r else functools.reduce(lambda x,y: x.e(op, y), lbs[1:], lbs[0])
      print(f"{out_lbs=}, {out_real=}")
      return MultiLazyBuffer(out_lbs, self.axis, out_real)
    assert all_same([x.device for x in msrcs]), f"all buffers must have the same device {[x.device for x in msrcs]}"

    # NOTE: they all have to share an axis, we always choose [-1]
    axis = axes[-1] if len(axes := dedup([x.axis for x in msrcs if x.axis is not None])) else None
    srcs = []
    for mlb in msrcs:
      if mlb.axis == axis: srcs.append(mlb.lbs)
      elif mlb.axis is None and axis is not None: srcs.append(to_sharded(mlb.lbs, axis))
      else: srcs.append(to_sharded([mlb.copy_to_device(lb.device) for lb in mlb.lbs], axis))
    return MultiLazyBuffer([lsrcs[0].e(op, *lsrcs[1:], arg=arg) for lsrcs in zip(*srcs)], axis)

  def _shape_to_single_shard(self, shape:Tuple[sint, ...], lb:LazyBuffer) -> Tuple[sint, ...]:
    return tuple(lb.shape[self.axis] if a == self.axis else s for a,s in enumerate(shape))

  def r(self, op:ReduceOps, new_shape:Tuple[sint, ...]) -> MultiLazyBuffer:
    if self.axis is not None and new_shape[self.axis] == 1:
      if not all(self.real):
        assert sum(self.real) == 1, self.real
        # just r individual lb
        ret = MultiLazyBuffer([x.r(op, new_shape) for x,r in zip(self.lbs,self.real)], self.axis, self.real)
        return ret
      # all-reduce on sharded axes
      return MultiLazyBuffer(all_reduce(op, [x.r(op, new_shape) for x in self.lbs]), None, self.real)
    # reduce on non sharded axes, piecewise is fine. if axis is None this is also correct
    return MultiLazyBuffer([x.r(op, self._shape_to_single_shard(new_shape, x)) for x in self.lbs], self.axis, self.real)

  # *** movement ops ***

  def reshape(self, arg:Tuple[sint, ...]):
    if self.axis is None: return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], None)
    if not all(self.real): return MultiLazyBuffer([x.reshape(arg) for x in self.lbs], self.axis, self.real)
    # TODO: this can be wrong
    st = ShapeTracker.from_shape(self.shape)
    rs = st.real_strides()[self.axis]
    new_axis = st.reshape(arg).real_strides().index(rs)
    return MultiLazyBuffer(
      [x.reshape(tuple(x.shape[self.axis] if a == new_axis else s for a,s in enumerate(arg))) for x in self.lbs], new_axis, self.real)

  def pad(self, arg:Tuple[Tuple[sint, sint], ...]):
    # assert self.axis is None or arg[self.axis] == (0,0), "padding not supported on sharded axis"
    if not all(self.real):
      # passthrough
      # TODO: this should pad zero tensors though so the backward is a shrink
      return self
    return MultiLazyBuffer([x.pad(arg) for x in self.lbs], self.axis, self.real)
  def expand(self, arg:Tuple[sint, ...]):
    if not all(self.real): return MultiLazyBuffer([x.expand(arg) for x in self.lbs], self.axis, self.real)
    # NOTE: this assert isn't needed, sharded axis can have dim 1
    assert self.axis is None or arg[self.axis] == self.shape[self.axis], "expand not supported on sharded axis"
    return MultiLazyBuffer([x.expand(self._shape_to_single_shard(arg, x)) for x in self.lbs], self.axis, self.real)
  def permute(self, arg:Tuple[int, ...]):
    # all permutes supported!
    return MultiLazyBuffer([x.permute(arg) for x in self.lbs], arg.index(self.axis) if self.axis is not None else None, self.real)
  def shrink(self, arg:Tuple[Tuple[sint, sint], ...]):
    assert self.axis is None or arg[self.axis] == (0, self.shape[self.axis]) or arg[self.axis] in self.bounds, \
      f"shrinking not supported {arg[self.axis]=}, {self.bounds=}"
    if self.axis is not None and arg[self.axis] in self.bounds:
      assert all(self.real), "cannot shrink on axis twice"
      # TODO: assert all other lbs are not touched
      idx = self.bounds.index(arg[self.axis])
      # print(f"{[i==idx for i in range(len(self.lbs))]=}")
      return MultiLazyBuffer(self.lbs, self.axis, [i==idx for i in range(len(self.lbs))])
    return MultiLazyBuffer(
      [x.shrink(tuple((0, x.shape[self.axis]) if i == self.axis else a for i,a in enumerate(arg))) for x in self.lbs], self.axis, self.real)
  def stride(self, arg:Tuple[int, ...]):
    assert self.axis is None or arg[self.axis] == 1, "flipping not supported on sharded axis"
    return MultiLazyBuffer([x.stride(arg) for x in self.lbs], self.axis, self.real)
