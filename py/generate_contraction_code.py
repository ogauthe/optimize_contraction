#!/usr/bin/env python3
import numpy as np
import sympy as sp
import re

tofill = '...'
regex = re.compile('[^a-zA-Z0-9]')

class AbstractTensor(object):
  """
  Class for abstract tensor. Each tensor has a shape (that can include formal
  variables), a name used to print it and a list of legs that can match other
  tensors. Two tensors can be contracted along a common leg.
  """

  def __init__(self,name,shape,legs):
    if len(shape) != len(legs):
      raise ValueError('shape and legs must have same length')
    self._name = name
    self._ndim = len(legs)
    self._shape = list(shape)
    self._legs = list(legs)
    self._size = np.prod(shape)

  @property
  def name(self):
    return self._name

  @property
  def shape(self):
    return self._shape

  @property
  def legs(self):
    return self._legs

  @property
  def ndim(self):
    return self._ndim

  @property
  def size(self):
    return self._size

  def __repr__(self):
    return self._name

  def raw_name(self):
    return regex.sub('',self._name)

def find_common_legs(A,B):
  return tuple(set(A.legs).intersection(B.legs))

def have_common_legs(A,B):
  return bool(find_common_legs(A,B))

def abstract_contraction(A,B,legs=None):
  if legs is None:
    legs = find_common_legs(A,B)
  if not legs:   # explicit exception, clearer than A.legs.index(l) one
    raise ValueError('Tensor have no common leg')
  legsA = [A.legs.index(l) for l in legs]
  legsB = [B.legs.index(l) for l in legs]
  axA = [k for k in range(A.ndim) if k not in legsA]
  axB = [k for k in range(B.ndim) if k not in legsB]
  name = "[" + A.name + '-' + B.name + ']'
  shape = [A.shape[i] for i in axA] + [B.shape[i] for i in axB]
  legs = [A.legs[i] for i in axA] + [B.legs[i] for i in axB]
  res = AbstractTensor(name,shape,legs)
  cpu = res.size
  for i in legsA:   # cpu cost = loop on returned shape
    cpu *= A.shape[i]  # + loop on every contracted leg
  mem = max(A.size,B.size,res.size)
  return AbstractTensor(name,shape,legs), (cpu,mem)


class TensorNetwork(object):
  """
  A class for abstract tensor network. Consists in a list of tensors that can
  be contracted and a list of previously contracted legs. Store the cpu cost of
  each contraction and the memory cost of each past state.
  """

  def __init__(self, *tensors):
    self._tensors = list(tensors)
    self._cpu = 0
    self._mem = sum(T.size for T in tensors)
    self._contracted = []
    self._ntens = len(tensors)

  @property
  def tensors(self):
    return self._tensors

  @property
  def ntens(self):
    return self._ntens

  @property
  def cpu(self):
    return self._cpu

  @property
  def mem(self):
    return self._mem

  @property
  def contracted(self):
    return self._contracted

  def copy(self):
    return TensorNetwork(*self._tensors, cpu=self._cpu, mem=self._mem,
                          contracted=self._contracted)

  def __repr__(self):
    return ','.join([T.name for T in self._tensors])


  def contract_legs(self,legs):
    tens = []
    i = self._ntens - 1
    mem0 = sum(T.size for T in self._tensors)
    while len(tens) != 2:
      if legs[0] in self._tensors[i].legs:
        tens.append(self.tensors.pop(i))
      i -= 1
    contracted, (cpu,mem) = abstract_contraction(*tens,legs=legs)
    self._tensors.append(contracted)
    mem += mem0
    self._cpu += cpu
    self._mem = max(self._mem,mem)
    self._ntens -= 1
    print(f'{tens[0].raw_name()} = {tens[0].raw_name()}.tranpose({tofill}).reshape({tofill})')
    print(f'{tens[1].raw_name()} = {tens[1].raw_name()}.tranpose({tofill}).reshape({tofill})')
    print(f'{contracted.raw_name()} = np.tensordot({tens[0].raw_name()},{tens[1].raw_name()},({tofill}),({tofill}))')
    print(f'del {tens[0].raw_name()}, {tens[1].raw_name()}')
    print(f'{contracted.raw_name()} = {contracted.raw_name()}.reshape({tofill})')

chi = 20
D = 3
#   C-0-T- -1
#   |   |
#   1   2
#   |   |
#   T-3-E- -2
#   |   |
#  -3   -4

C = AbstractTensor('C',(chi,chi),(0,1))
T1 = AbstractTensor('T1',(chi,D**2,chi),(0,2,-1))
T2 = AbstractTensor('T2',(chi,chi,D**2),(1,-3,3))
E = AbstractTensor('E',(D**2,D**2,D**2,D**2),(2,3,-4,-2))
sequence = [(0,),(1,),(2,3)]
print(f'test: symmetric CTMRG contraction scheme, chi={chi}, D={D}')

tn = TensorNetwork(C,T1,T2,E)
for legs in sequence:
  tn.contract_legs(legs)

print(tn, ': cpu = ', tn.cpu, ', mem = ', tn.mem, sep='')


