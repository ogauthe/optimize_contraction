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

  def __init__(self, *tensors, cpu=[], mem=None, contractions=[]):
    self._tensors = list(tensors)
    self._cpu = cpu.copy()
    self._contractions = contractions.copy()
    self._ntens = len(tensors)
    if mem is None:
      self._mem = [sum(T.size for T in tensors)]
    else:
      self._mem = mem.copy()

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
  def contractions(self):
    return self._contractions

  def copy(self):
    return TensorNetwork(*self._tensors, cpu=self._cpu, mem=self._mem,
                          contractions=self._contractions)

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
    """
    Two TN are equal if they consist in exactly the same tensors contracted
    the same way.
    """
    names = sorted(t.name for t in self._tensors)
    tn_names = sorted(t.name for t in tn._tensors)
    return names == tn_names

  def contract(self,i,j):
    """
    contract tensors i and j along every common legs
    """
    legs = find_common_legs(self._tensors[i],self._tensors[j])
    T,cpu = abstract_contraction(self._tensors[i],self._tensors[j],legs)
    self._cpu.append(cpu)
    self._mem.append(sum(t.size for t in self._tensors)+T.mem)# store i,j and T
    del self._tensors[max(i,j)], self._tensors[min(i,j)]
    self._tensors.append(T)
    self._ntens -= 1
    self._contractions.append(tuple(legs))


# Always contract every compatible legs of two tensors. This is not the same
# as choosing one leg and contracting it at every step.
def bruteforce_contraction(*tensors, compare=True):
  tn = TensorNetwork(*tensors)
  if len(tensors) < 2:
    return [tn]
  queue = [tn]
  contracted = []
  while queue:
    tn = queue.pop(0)
    for i in range(tn.ntens-1):
      for j in range(i+1,tn.ntens):
        if have_common_legs(tn.tensors[i],tn.tensors[j]):
          new_tn = tn.copy()
          new_tn.contract(i,j)
          if new_tn.ntens == 1:
            contracted.append(new_tn)
          else:
            c_legs = set(chain.from_iterable(new_tn.contractions))
            append = True
            rm = []
            for k,tnk in enumerate(queue): # remove dupplicate and worse path
              if set(chain.from_iterable(tnk.contractions)) == c_legs:
                if (compare and tnk < new_tn) or tnk == tn:
                  append = False
                  break
                elif compare and new_tn < tnk:
                  rm.append(k)
            if append:
              for k in rm[::-1]:
                del queue[k]
              queue.append(new_tn)

  def keep_it(c,c2):
    return not ((c==c2 and (not (c is c2)) ) or (compare and c<c2))

  for c in contracted:  # remove bad ones
    contracted[:] = [c2 for c2 in contracted if keep_it(c,c2)]
  return contracted


chi = 100
D = 3
print('test: symmetric CTMRG contraction scheme, chi={chi}, D={D}')
print('The results must be [E-[T1-[C-T2]]] and [E-[T2-[C-T1]]]')
C = AbstractTensor('C',(chi,chi),(2,0))
T1 = AbstractTensor('T1',(chi,chi,D**2),(0,1,3))
T2 = AbstractTensor('T2',(chi,chi,D**2),(2,6,4))
E = AbstractTensor('E',(D**2,D**2,D**2,D**2),(3,4,7,5))

res = bruteforce_contraction(C,T1,T2,E)
print('optimal schemes =', res)
print(res[0], ': cpu = ', sum(res[0].cpu), ', mem = ', max(res[0].mem), sep='')

print('\ntest with formal variables')
import sympy as sp
chif,Df = sp.symbols('chi,D')
Cf = AbstractTensor('C',(chif,chif),(2,0))
T1f = AbstractTensor('T1',(chif,chif,Df**2),(0,1,3))
T2f = AbstractTensor('T2',(chif,chif,Df**2),(2,6,4))
Ef = AbstractTensor('E',(Df**2,Df**2,Df**2,Df**2),(3,4,7,5))
resf =  bruteforce_contraction(Cf,T1f,T2f,Ef,compare=False) # cannot compare formal
print(resf[1], ': cpu = ', sum(resf[1].cpu), '\nmem = max(', resf[1].mem, ')', sep='')
