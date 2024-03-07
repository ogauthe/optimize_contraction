#!/usr/bin/env python3
import numpy as np
import sympy as sp
import re
import json
from sys import argv


class AbstractTensor:
    """
    Class for abstract tensor. Each tensor has a shape (that can include formal
    variables), a name used to print it and a list of legs that can match other
    tensors. Two tensors can be contracted along a common leg.
    """

    regex = re.compile("[^a-zA-Z0-9_]")

    def __init__(self, name, legs, shape, n_row_leg, initial=False):
        if len(shape) != len(legs):
            raise ValueError("shape and legs must have same length")
        self._name = name
        self._legs = list(legs)
        self._shape = list(shape)
        self._n_row_leg = int(n_row_leg)
        self._initial = bool(initial)
        self._ndim = len(legs)
        self._size = np.prod(shape)

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, name):
        self._name = name

    @property
    def legs(self):
        return self._legs

    @property
    def shape(self):
        return self._shape

    @property
    def initial(self):
        return self._initial

    @property
    def ndim(self):
        return self._ndim

    @property
    def n_row_leg(self):
        return self._n_row_leg

    @property
    def size(self):
        return self._size

    def __repr__(self):
        return self._name

    def raw_name(self):
        return self.regex.sub("", self._name)


def find_common_legs(A, B):
    return tuple(set(A.legs).intersection(B.legs))


def have_common_legs(A, B):
    return bool(find_common_legs(A, B))


def abstract_contraction(A, B, legs=None):
    if legs is None:
        legs = find_common_legs(A, B)
    if not legs:  # explicit exception, clearer than A.legs.index(l) one
        raise ValueError("Tensor have no common leg")
    legsA = [A.legs.index(leg) for leg in legs]
    legsB = [B.legs.index(leg) for leg in legs]
    axA = [k for k in range(A.ndim) if k not in legsA]
    axB = [k for k in range(B.ndim) if k not in legsB]
    name = "[" + A.name + "-" + B.name + "]"
    shape = [A.shape[i] for i in axA] + [B.shape[i] for i in axB]
    legs = [A.legs[i] for i in axA] + [B.legs[i] for i in axB]
    res = AbstractTensor(name, legs, shape, A.n_row_leg)
    cpu = res.size
    for i in legsA:  # cpu cost = loop on returned shape
        cpu *= A.shape[i]  # + loop on every contracted leg
    mem = A.size + B.size + res.size  # unreachable upper bound, cannot get max
    return res, (cpu, mem)


class TensorNetwork:
    """
    A class for abstract tensor network. Consists in a list of tensors that can
    be contracted and a list of previously contracted legs. Store the cpu cost of
    each contraction and the memory cost of each past state.
    """

    def __init__(self, tensors):
        self._tensors = list(tensors)
        self._cpu = 0
        self._mem = [sum(T.size for T in tensors)]
        self._contracted = []
        self._n_tensors = len(tensors)

    @property
    def tensors(self):
        return self._tensors

    @property
    def n_tensors(self):
        return self._n_tensors

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
        return TensorNetwork(
            self._tensors,
            cpu=self._cpu,
            mem=self._mem.copy(),
            contracted=self._contracted,
        )

    def __repr__(self):
        return ",".join([T.name for T in self._tensors])

    def contract_legs(self, legs):
        tens = []
        i = self._n_tensors - 1
        mem0 = sum(T.size for T in self._tensors)
        while len(tens) != 2:
            if legs[0] in self._tensors[i].legs:
                tens.append(self.tensors.pop(i))
            i -= 1
        contracted, (cpu, mem) = abstract_contraction(tens[0], tens[1], legs=legs)
        self._tensors.append(contracted)
        self._cpu += cpu
        self._mem.append(mem + mem0)
        self._n_tensors -= 1

    def contract_and_generate_code(self, legs):
        # 1. find tensors A and B that have legs to contract
        tens = []
        i = self._n_tensors - 1
        mem0 = sum(T.size for T in self._tensors)
        while len(tens) != 2:
            if legs[0] in self._tensors[i].legs:
                tens.append(self.tensors.pop(i))
            i -= 1

        # 2. find legs indices in A and B
        A, B = tens
        legsA = tuple(A.legs.index(leg) for leg in legs)  # indices of legs to contract
        legsB = tuple(B.legs.index(leg) for leg in legs)  # indices of legs to contract
        axA = tuple(
            k for k in range(A.ndim) if k not in legsA
        )  # indices of A other legs
        axB = tuple(
            k for k in range(B.ndim) if k not in legsB
        )  # indices of B other legs
        permA = (axA, legsA)
        permB = (legsB, axB)

        # 3. find contracted tensor features
        if self._n_tensors == 2:
            ABname = "out"
        elif A.initial:
            if B.initial:
                ABname = f"_tmp{i}"
            else:
                ABname = B.raw_name()
        else:
            ABname = A.raw_name()
        ABshape = [A.shape[i] for i in axA] + [B.shape[i] for i in axB]
        ABlegs = [A.legs[i] for i in axA] + [B.legs[i] for i in axB]
        AB = AbstractTensor(ABname, ABlegs, ABshape, len(axA))
        cpu = AB.size
        for i in legsA:  # cpu cost = loop on returned shape
            cpu *= A.shape[i]  # + loop on every contracted leg
        mem = mem0 + A.size + B.size + AB.size

        # 4. generate code
        trivial = (tuple(range(A.n_row_leg)), tuple(range(A.n_row_leg, A.ndim)))
        if permA == trivial:
            newA = A.raw_name()
        else:
            if A.initial:
                newA = A.raw_name() + "p"
            else:
                newA = A.raw_name()
            if permA == trivial[::-1]:  # matrix transpose
                print(f"{newA} = {A.raw_name()}.transpose()")
            else:
                print(f"{newA} = {A.raw_name()}.permute{permA}")
        trivial = (tuple(range(B.n_row_leg)), tuple(range(B.n_row_leg, B.ndim)))
        if permB == trivial:
            newB = B.raw_name()
        else:
            if B.initial:
                newB = B.raw_name() + "p"
            else:
                newB = B.raw_name()
            if permB == trivial[::-1]:  # matrix transpose
                print(f"{newB} = {B.raw_name()}.transpose()")
            else:
                print(f"{newB} = {B.raw_name()}.permute{permB}")
        print(f"{AB.raw_name()} = {newA} @ {newB}")
        self._tensors.append(AB)
        self._cpu += cpu
        self._mem.append(mem)
        self._n_tensors -= 1


if len(argv) < 2:
    input_file = "input_sample_gen_py.json"
    print("\nNo input file given, use", input_file)
else:
    input_file = argv[1]
    print("\nTake input parameters from file", input_file)


with open(input_file) as fin:
    input_data = json.load(fin)

sequence = input_data["sequence"]
input_tensors = []
for t0 in input_data["tensors"]:
    sh0 = sp.sympify(t0["shape"])
    sh = []
    for d in sh0:
        try:
            d = int(d)
        except TypeError:
            pass
        sh.append(d)
    t = AbstractTensor(t0["name"], t0["legs"], sh, t0["n_row_leg"], initial=True)
    input_tensors.append(t)

legs_map = {}
var = {}
for t in input_tensors:
    for i, (leg, d) in enumerate(zip(t.legs, t.shape)):
        if t.legs.count(leg) > 1:
            raise ValueError(
                f"Tensor {t.name} has twice the same leg. Trace is not allowed."
            )
        if leg in legs_map.keys():
            if not legs_map[leg][0]:
                raise ValueError(f"Leg {leg} appears more than twice")
            if legs_map[leg][1] != d:
                raise ValueError(f"Leg {leg} has two diffent dimensions")
            legs_map[leg] = (False, d)  # once, dim
        else:
            legs_map[leg] = (True, d)  # once, dim
            if isinstance(d, sp.Basic):
                var[d] = (t, i)


print("Input tensors:")
for t in input_tensors:
    print(f"name: {t.name}, legs: {t.legs}, shape: {t.shape}")
print("Contraction sequence:", sequence)

print()
tn = TensorNetwork(input_tensors)
for legs in sequence:
    tn.contract_and_generate_code(legs)

if tn.n_tensors != 1:
    raise ValueError("Final number of tensors is not 1")
final = tn.tensors[0]
print(f"# exit tensor: {final} with name {final.raw_name()} and legs {final.legs}")
order = tuple(np.argsort(np.abs(final.legs)))
if order != tuple(range(final.ndim)):
    print(f"# reorder with: {final.raw_name()} = {final.raw_name()}.permute{order}")

print(
    f"\nresult: {tn}",
    f"total cpu: {sp.factor(tn.cpu)}",
    f"mem by step: {sp.factor(tn.mem)}",
    sep="\n",
)
