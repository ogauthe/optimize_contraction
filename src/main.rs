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

#[derive(Debug, Clone)]
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
    for (i,ti) in self.tensors.iter().enumerate() {
      for (j,tj) in self.tensors[i+1..].iter().enumerate() {
        let legs = ti & tj;
        if legs != 0 {  // if tensors have common leg
          let mut child_tensors = self.tensors.clone();
          child_tensors.remove(i+1+j);
          child_tensors.remove(i);
          let ti_dot_tj = ti^tj;
          child_tensors.push(ti_dot_tj);
          let mem = self.tensors.iter().map(|t| size(*t,self.leg_dim)).sum::<u64>() + size(*ti,self.leg_dim) + size(*tj,self.leg_dim) + size(ti_dot_tj, self.leg_dim);
          let child = TensorNetwork {
            cpu: self.cpu + size(ti|tj,self.leg_dim),
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

fn bruteforce_contraction(tn0: TensorNetwork, n_c:u64) -> Vec<u8> {
  //let n_l = tn0.tensors.iter().map(|t| 64-t.leading_zeros()).max().unwrap();  // total number of leg != number of legs to contract
  let N = 1usize<<n_c;
  let mut vec:Vec<usize> = (0..N).collect();
  vec.sort_by_key(|c| c.count_ones());
  let mut tn_vec:Vec<TensorNetwork> = vec![TensorNetwork { cpu:u64::max_value(), mem:0, contracted:0, parent:0, leg_dim:tn0.leg_dim, tensors: Vec::new()};N];
  tn_vec[0] = tn0;
  for &n in vec.iter() {
    if tn_vec[n].cpu < u64::max_value() {   // some states are not reachable
      for child in tn_vec[n].generate_children() {
        if child.cpu < tn_vec[child.contracted as usize].cpu {
          tn_vec[child.contracted as usize] = child.clone();   // need to clone tensors
        }
      }
    }
  }

  vec![0,1,2]
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
  println!("{}",tn0.contracted);

  let legs = bruteforce_contraction(tn0,4);
  println!("{:?}", legs);

  println!("===========   Completed   =========");
}
