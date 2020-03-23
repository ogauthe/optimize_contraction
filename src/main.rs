/*
#[derive(Debug)]
struct AbstractTensor {
  legs: u64,
  leg_dim: &Vec<u64>,
}

impl AbstractTensor {
  pub fn new(legs: u64, leg_dim: &Vec<u64>) {
    AbstractTensor {
      legs,
      leg_dim,
    }
  }

  fn size(&self) -> u64 {
    let mut s = 1u64;
    for (i, d) in self.leg_dim.iter().enumerate() {
      if (self.legs << i)%2 as bool {
        s *= d;
      }
    }
    s
  }
}
*/


fn size(legs: u64, leg_dim: &Vec<u64>) -> u64 {
  let mut s = 1u64;
  for (i, d) in leg_dim.iter().enumerate() {
    if (legs >> i)%2 != 0 {
      s *= d;
    }
  }
  s
}

#[derive(Debug)]
struct TensorNetwork<'a> {
  cpu: u64,
  mem: u64,
  contracted: u64,
  parent: u64,
  leg_dim: &'a Vec<u64>,
  tensors: Vec<u64>,
}


impl TensorNetwork<'_> {
  pub fn new(tensors: Vec<u64>, leg_dim: &Vec<u64>) -> TensorNetwork {
    TensorNetwork {
      cpu: 0,
      mem: tensors.iter().map(|x| size(*x,leg_dim)).sum(),
      contracted: 0,
      parent: 0,
      leg_dim,
      tensors,
    }
  }

  fn generate_children(&self) -> Vec<TensorNetwork> {
    let mut children = Vec::new();
    for i in 0..self.tensors.len()-1 {
      for j in i+1..self.tensors.len() {
        let legs = self.tensors[i] & self.tensors[j];
        if legs != 0 {  // if tensors have common leg
          let mut child_tensors = self.tensors.clone();
          child_tensors.remove(j);
          child_tensors.remove(i);
          let ti_dot_tj = self.tensors[i] ^ self.tensors[j];
          child_tensors.push(ti_dot_tj);
          let mem = self.tensors.iter().map(|x| size(*x,self.leg_dim)).sum::<u64>() + size(self.tensors[i],self.leg_dim) + size(self.tensors[i],self.leg_dim) + size(ti_dot_tj, self.leg_dim);
          let child = TensorNetwork {
            cpu: self.cpu + size(self.tensors[i]|self.tensors[j],self.leg_dim),
            mem: std::cmp::max(self.mem,mem),
            contracted: self.contracted | legs,
            parent: self.contracted,
            leg_dim: self.leg_dim,
            tensors: child_tensors,
          };
          children.push(child);
        }
      }
    }
  children
  }
}

fn main() {
  println!("=============   Begin   ===========");

  /*   C-0-T- -1
   *   |   |
   *   1   2
   *   |   |
   *   T-3-E- -2
   *   |   |
   *  -3   -4
   */
  let leg_dim = vec![20,20,9,9,20,9,20,9];
  let tensors = vec![3,21,74,172];

  let tn0 = TensorNetwork::new(tensors, &leg_dim);
  println!("{:?}",tn0);
  let children = tn0.generate_children();
  println!("{:?}",children);
  let tn1 = &children[0];
  let tn2 = &tn1.generate_children()[0];
  println!("{:?}",tn2);
  let children3 = tn2.generate_children();
  println!("{:?}",children3);

  println!("===========   Completed   =========");
}
