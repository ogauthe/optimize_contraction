#include <string>
#include <iostream>
#include <tuple>
#include <vector>

class AbstractTensor
{
  public:
    AbstractTensor(std::string const &name, std::vector<unsigned> const &shape, std::vector<int> const &legs);
    ~AbstractTensor();

  //std::string get_name() const { return _name; }
  std::string const& get_name() const { return _name; }
  //std::vector<unsigned> get_shape() const { return _shape; }
  std::vector<unsigned> const& get_shape() const { return _shape; }
  //std::vector<int> get_legs() const { return _legs; }
  std::vector<int> const& get_legs() const { return _legs; }
  unsigned get_ndim() const { return _ndim; }
  unsigned long get_size() const { return _size; }
  std::vector<int> find_common_legs(AbstractTensor const &t) const;
  bool has_common_legs(AbstractTensor const &t) const;
  std::tuple<AbstractTensor,unsigned long> dot(AbstractTensor const &t, std::vector<int> const &contracted_legs_) const;

  friend std::ostream & operator << (std::ostream &out, AbstractTensor const &t);

  private:
  std::string _name;
  std::vector<unsigned> _shape;
  std::vector<int> _legs;
  unsigned _ndim;
  unsigned long _size;
};
