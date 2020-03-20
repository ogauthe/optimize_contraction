#!/usr/bin/env python3
import numpy as np
from itertools import chain

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
  name = '[' + A.name + '-' + B.name + ']'
  shape = [A.shape[i] for i in axA] + [B.shape[i] for i in axB]
  legs = [A.legs[i] for i in axA] + [B.legs[i] for i in axB]
  res = AbstractTensor(name,shape,legs)
  cpu = res.size
  for i in legsA:   # cpu cost = loop on returned shape
    cpu *= A.shape[i]  # + loop on every contracted leg
  return AbstractTensor(name,shape,legs), cpu


class TensorNetwork(object):
  """
  A class for abstract tensor network. Consists in a list of tensors that can
  be contracted and a list of previously contracted legs. Store the cpu cost of
  each contraction and the memory cost of each past state.
  """

  def __init__(self, *tensors, cpu=0, mem=0, contracted=0):
    self._tensors = list(tensors)
    self._cpu = cpu
    self._contracted = contracted
    self._ntens = len(tensors)
    if mem == 0:
      self._mem = sum(T.size for T in tensors)
    else:
      self._mem = mem

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

  def __lt__(self,tn):
    """
    A < B if both the memory and cpu cost of A are lower. Do *NOT* check if
    current state is the same.
    """
    mem = max(self._mem)
    tn_mem = max(tn._mem)
    cpu = sum(self._cpu)
    tn_cpu = sum(tn.cpu)
    return (mem <= tn_mem and cpu < tn_cpu) or (mem < tn_mem and cpu <= tn_cpu)

  def __gt__(self,tn):
    """
    This is not the same as not(tn<self), since both can be False.
    """
    mem = max(self._mem)
    tn_mem = max(tn._mem)
    cpu = sum(self._cpu)
    tn_cpu = sum(tn.cpu)
    return (mem >= tn_mem and cpu > tn_cpu) or (mem > tn_mem and cpu >= tn_cpu)

  def __eq__(self,tn):
    return self._contracted == tn.contracted

  def contract(self,i,j):
    """
    contract tensors i and j along every common legs
    """
    legs = find_common_legs(self._tensors[i],self._tensors[j])
    T,cpu = abstract_contraction(self._tensors[i],self._tensors[j],legs)
    self._cpu += cpu
    self._mem = max(self._mem,(sum(t.size for t in self._tensors)+T.size))# store i,j and T
    del self._tensors[max(i,j)], self._tensors[min(i,j)]
    self._tensors.append(T)
    self._ntens -= 1
    self._contracted |= sum(2**l for l in legs)

  def generate_children(self):
    children = []
    for i in range(self._ntens-1):
      for j in range(i+1,self._ntens):
        if have_common_legs(self._tensors[i],self._tensors[j]):
          child = self.copy()
          child.contract(i,j)
          children.append(child)
    return children


def popcount(n):
  return bin(n).count('1')

cputype = np.uint64
maxcpu = np.iinfo(cputype).max

def bruteforce_contraction(*tensors):
  c_legs = []
  o_legs = []
  for t in tensors:
    for l in t.legs:
      if l < 0:
        o_legs.append(l)
      else:
        c_legs.append(l)
  if sorted(o_legs) != list(range(min(o_legs),0)):
    raise Exception('every legs < 0 must appear once and only once')
  if (np.bincount(c_legs) - 2).any():
    raise Exception('every legs >= 0 must appear exactly twice')

  n_c = len(c_legs)//2
  c_legs = np.arange(n_c)
  base2 = 2**np.arange(n_c)
  vecs = [[] for i in range(n_c)]
  for i in range(2**n_c-1):
     vecs[popcount(i)].append(i)
  TN_L = [TensorNetwork(cpu=maxcpu)]*(2**n_c)
  TN_L[0] = TensorNetwork(*tensors)
  for n in range(n_c):
    for tn_ID in vecs[n]:
      mother = TN_L[tn_ID]
      if mother.cpu < maxcpu:
        for child in mother.generate_children():
          if child.cpu < TN_L[child.contracted].cpu:
            TN_L[child.contracted] = child
  return TN_L[-1]



chi = 100
D = 4
#
#  C-0-T- -1
#  |   |
#  1   2
#  |   |
#  T-3-E- -2
#  |   |
#  -3  -4
C = AbstractTensor('C',(chi,chi),(1,0))
T1 = AbstractTensor('T1',(chi,chi,D**2),(0,2,-1))
T2 = AbstractTensor('T2',(chi,chi,D**2),(1,-3,3))
E = AbstractTensor('E',(D**2,D**2,D**2,D**2),(2,3,-4,-2))
print(f'test: symmetric CTMRG contraction scheme, chi={chi}, D={D}')
print('The result must be [E-[T1-[C-T2]]] or [E-[T2-[C-T1]]]')

res = bruteforce_contraction(C,T1,T2,E)
print('optimal schemes =', res)
print(res, ': cpu = ', res.cpu, ', mem = ', res.mem, sep='')
