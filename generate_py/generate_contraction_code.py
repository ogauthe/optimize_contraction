#!/usr/bin/env python3
import numpy as np
import sympy as sp
import re
import json
from sys import argv



class AbstractTensor(object):
  """
  Class for abstract tensor. Each tensor has a shape (that can include formal
  variables), a name used to print it and a list of legs that can match other
  tensors. Two tensors can be contracted along a common leg.
  """
  regex = re.compile('[^a-zA-Z0-9_]')

  def __init__(self,name,legs,shape,todel=True):
    if len(shape) != len(legs):
      raise ValueError('shape and legs must have same length')
    self._name = name
    self._legs = list(legs)
    self._shape = list(shape)
    self._todel = todel    # del tensor unless starting one
    self._ndim = len(legs)
    self._size = np.prod(shape)

  @property
  def name(self):
    return self._name

  @name.setter
  def name(self,name):
    self._name = name

  @property
  def legs(self):
    return self._legs

  @property
  def shape(self):
    return self._shape

  @property
  def todel(self):
    return self._todel

  @todel.setter
  def todel(self,todel):
    self._todel = todel

  @property
  def ndim(self):
    return self._ndim

  @property
  def size(self):
    return self._size

  def __repr__(self):
    return self._name

  def raw_name(self):
    return self.regex.sub('',self._name)

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
  res = AbstractTensor(name,legs,shape)
  cpu = res.size
  for i in legsA:   # cpu cost = loop on returned shape
    cpu *= A.shape[i]  # + loop on every contracted leg
  mem = A.size + B.size + res.size  #unreachable upper bound, cannot get max
  return res, (cpu,mem)


class TensorNetwork(object):
  """
  A class for abstract tensor network. Consists in a list of tensors that can
  be contracted and a list of previously contracted legs. Store the cpu cost of
  each contraction and the memory cost of each past state.
  """

  def __init__(self, *tensors):
    self._tensors = list(tensors)
    self._cpu = 0
    self._mem = [sum(T.size for T in tensors)]
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
    return TensorNetwork(*self._tensors, cpu=self._cpu, mem=self._mem.copy(),
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
    contracted, (cpu,mem) = abstract_contraction(tens[0],tens[1],legs=legs)
    self._tensors.append(contracted)
    self._cpu += cpu
    self._mem.append(mem+mem0)
    self._ntens -= 1

  def contract_and_generate_code(self,legs):
    # 1. find tensors A and B that have legs to contract
    tens = []
    i = self._ntens - 1
    mem0 = sum(T.size for T in self._tensors)
    while len(tens) != 2:
      if legs[0] in self._tensors[i].legs:
        tens.append(self.tensors.pop(i))
      i -= 1

    # 2. find legs indices in A and B
    A,B = tens
    legsA = [A.legs.index(l) for l in legs]  # indices of legs to contract in A
    legsB = [B.legs.index(l) for l in legs]  # indices of legs to contract in B
    axA = [k for k in range(A.ndim) if k not in legsA]    # indices of A other legs
    axB = [k for k in range(B.ndim) if k not in legsB]    # indices of B other legs

    # 3. find contracted tensor features
    ABname = "[" + A.name + '-' + B.name + ']'
    ABshape = [A.shape[i] for i in axA] + [B.shape[i] for i in axB]
    ABlegs = [A.legs[i] for i in axA] + [B.legs[i] for i in axB]
    AB = AbstractTensor(ABname,ABlegs,ABshape)
    cpu = AB.size
    for i in legsA:   # cpu cost = loop on returned shape
      cpu *= A.shape[i]  # + loop on every contracted leg
    mem = mem0 + A.size + B.size + AB.size

    # 4. generate code
    Aupdate = transpose_reshape(A,axA,legsA)
    Bupdate = transpose_reshape(B,legsB,axB)
    print(f'{AB.raw_name()} = np.dot({Aupdate},{Bupdate})'+ (AB.ndim > 1)*f'.reshape{tuple(ABshape)}')
    if A.todel or B.todel:
      print('del ' + A.todel*(A.raw_name()+B.todel*", ") + B.todel*B.raw_name())
    #print(f'{contracted.raw_name()} = {contracted.raw_name()}.reshape({tofill})')
    self._tensors.append(AB)
    self._cpu += cpu
    self._mem.append(mem)
    self._ntens -= 1


def transpose_reshape(A,axA,legsA):
  orderA = tuple(axA + legsA)
  name = A.raw_name()
  sh = np.prod([A.shape[k] for k in axA]), np.prod([A.shape[k] for k in legsA])
  if orderA == tuple(range(len(orderA))):  # no transpose
    if A.ndim > 2:
      return f'{name}.reshape{sh}'
    return name   # (0,) or (0,1): nothing
  if A.ndim == 2:
    return f'{name}.T'
  if A.todel:
    updated_name = name
  else:
    A.name = A.name + '_'
    A.todel = True
  print(f'{A.raw_name()} = {name}.transpose{orderA}.reshape{sh}')
  return A.raw_name()

if len(argv) < 2:
  input_file = 'input_sample_gen_py.json'
  print("\nNo input file given, use", input_file)
else:
  input_file = argv[1]
  print('\nTake input parameters from file', input_file)



with open(input_file) as f:
    d = json.load(f)
    sequence = d['sequence']
    tensL = [AbstractTensor(t['name'],t['legs'],sp.sympify(t['shape']),False) for t in d['tensors']]

legs_map = {}
var = {}
for t in tensL:
  for i,(l,d) in enumerate(zip(t.legs,t.shape)):
    if t.legs.count(l) > 1:
      raise ValueError(f'Tensor {t.name} has twice the same leg. Trace is not allowed.')
    if l in legs_map.keys():
      if not legs_map[l][0]:
       raise ValueError(f"Leg {l} appears more than twice")
      if legs_map[l][1] != d:
       raise ValueError(f"Leg {l} has two diffent dimensions")
      legs_map[l] = (False,d)  # once, dim
    else:
      legs_map[l] = (True,d)   # once, dim
      var[d] = (t,i)

print("Tensors:")
for t in tensL:
  print(f'name: {t.name}, legs: {t.legs}, shape: {t.shape}')
print("sequence:", sequence)

print()
for (d,(t,i)) in var.items():
  print(f'{d} = {t}.shape[{i}]')

tn = TensorNetwork(*tensL)
for legs in sequence:
  tn.contract_and_generate_code(legs)

if tn.ntens != 1:
  raise ValueError('Final number of tensors is not 1')
final = tn.tensors[0]
print(f'# exit tensor: {final} with name {final.raw_name()} and legs {final.legs}')
order = tuple(np.argsort(np.abs(final.legs)))
if order != tuple(range(final.ndim)):
  print(f'# reorder with: {final.raw_name()} = {final.raw_name()}.transpose{order}.copy()')

print(f'\nresult: {tn}', f'total cpu: {sp.factor(tn.cpu)}', f'mem by step: {sp.factor(tn.mem)}', sep='\n')
