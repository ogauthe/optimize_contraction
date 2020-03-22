#include <string>
#include <iostream>
#include <vector>
#include <algorithm>    // std::find
#include <tuple>
#include "abstract_tensor.hpp"

AbstractTensor::AbstractTensor(std::string const &name, std::vector<unsigned> const &shape, std::vector<short> const &legs): _name(name), _shape(shape), _legs(legs), _ndim(legs.size()), _size(1)
{
  // should throw exception if legs and shape have different size
  for (auto i:shape) { _size *= i; }
}


AbstractTensor::~AbstractTensor()
{}

std::vector<short> AbstractTensor::find_common_legs(AbstractTensor const &t) const
{
  std::vector<short> legs;
  for (auto l1: _legs) {
    for (auto l2: t._legs) {
      if (l1==l2) {
        legs.push_back(l1);
        break;
      }
    }
  }
  return legs;
}

bool AbstractTensor::has_common_legs(AbstractTensor const &t) const
{
  bool b = bool(find_common_legs(t).size());
  return b;
}

std::ostream & operator << (std::ostream &out, const AbstractTensor &t)
{
    out << t._name;
    return out;
}

std::tuple<AbstractTensor,unsigned long> AbstractTensor::dot(AbstractTensor const &t, std::vector<short> const &contracted_legs_=std::vector<short>()) const
{
  std::vector<short> contracted_legs = contracted_legs_;
  if (contracted_legs.size()==0) {
    contracted_legs = find_common_legs(t);
  }
  // should throw exception if legs is empty
  std::string rname = '[' + _name + '-' + t._name + ']';
  std::vector<unsigned> rshape;
  std::vector<short> rlegs;

  for (unsigned i=0; i<_ndim; i++) {
    if (std::find(contracted_legs.begin(), contracted_legs.end(),_legs[i]) == contracted_legs.end()) {
      rshape.push_back(_shape[i]);
      rlegs.push_back(_legs[i]);
    }
  }

  unsigned long cpu = _size;   // CPU cost of contraction
  for (unsigned i=0; i<t._ndim; i++) {
    if (std::find(contracted_legs.begin(), contracted_legs.end(),t._legs[i]) == contracted_legs.end()) {
      rshape.push_back(t._shape[i]);
      rlegs.push_back(t._legs[i]);
      cpu *= t._shape[i];
    }
  }

  AbstractTensor res(rname,rshape,rlegs);
  return std::make_tuple(res,cpu);
}



