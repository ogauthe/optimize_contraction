#include "abstract_tensor.cpp"
#include <iostream>

int main(int argv, char* argv[])
{
  std::vector<int> legsA = {0,1,2}, legsB = {0,3,4};
  std::vector<unsigned> shapeA = {2,3,4}, shapeB = {2,5,6};
  AbstractTensor A("A",shapeA,legsA), B("B",shapeB,legsB);
  auto C = A.dot(B);
  std::cout << C << std::endl;

  return 0;
}
