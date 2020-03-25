type Dimension = u64;

#[derive(Debug, Clone)]
struct TensorNetwork {
  cpu: Dimension,
  mem: Dimension,
  contracted: usize,
  parent: usize,
  tensors: Vec<usize>,
}


impl TensorNetwork {
  pub fn new(tensors: Vec<usize>) -> TensorNetwork {
    TensorNetwork {
      cpu: 0,
      mem: tensors.iter().map(|x| TensorNetwork::measure(*x)).sum(),
      contracted: 0,
      parent: 0,
      tensors,
    }
  }

  fn measure(tensor: usize) -> Dimension {
    let mut s = 1;
    for (i, d) in leg_dim.iter().enumerate() {
      if (tensor >> i)%2 != 0 {
        s *= d;
      }
    }
    s
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
          let mem = self.tensors.iter().map(|t| TensorNetwork::measure(*t)).sum::<Dimension>() + TensorNetwork::measure(*ti) + TensorNetwork::measure(*tj) + TensorNetwork::measure(ti_dot_tj);
          let child = TensorNetwork {
            cpu: self.cpu + TensorNetwork::measure(ti|tj),
            mem: std::cmp::max(self.mem,mem),
            contracted: self.contracted | legs,
            parent: self.contracted,
            tensors: child_tensors,
          };
          children.push(child);
        }
      }
    }
  children
  }
}

fn bruteforce_contraction(tn0: TensorNetwork, n_c:usize) -> (Vec<usize>,Dimension,Dimension) {
  //let n_l = tn0.tensors.iter().map(|t| 64-t.leading_zeros()).max().unwrap();  // total number of leg != number of legs to contract
  let n = 1<<n_c;
  let mut vec:Vec<usize> = (0..n).collect();
  vec.sort_by_key(|c| c.count_ones());
  let mut tn_vec:Vec<TensorNetwork> = vec![TensorNetwork { cpu:Dimension::max_value(), mem:0, contracted:0, parent:0, tensors: Vec::new() };n];
  tn_vec[0] = tn0;
  for &n in vec.iter() {
    for child in tn_vec[n].generate_children() {
      if child.cpu < tn_vec[child.contracted].cpu {
        tn_vec[child.contracted] = child.clone();   // need to clone tensors
      }
    }
  }

  let mut sequence:Vec<usize> = Vec::new();
  let mut k = n-1;
  while k != 0 {
    sequence.push(tn_vec[k].contracted);
    k = tn_vec[k].parent;
  }
  sequence.reverse();
  (sequence, tn_vec[n-1].cpu, tn_vec[n-1].mem)
}

static leg_dim:[Dimension;8] = [20,20,9,9,20,9,20,9];
fn main() {
  println!("=============   Begin   ===========");

  /*   C-0-T- -1
   *   |   |
   *   1   2
   *   |   |
   *   T-3-E- -2
   *   |   |
   *  -3   -4
   *
   *  C: 3
   *  T1: 21
   *  T2: 74
   *  E: 172
   *
   * Solution: 0 -> (1 or 2) -> 3 -> 15
   *
   */

  let tensors = vec![3,21,74,172];

  let tn0 = TensorNetwork::new(tensors);
  println!("{:?}",tn0);

  let (sequence,cpu,mem) = bruteforce_contraction(tn0,4);
  println!("cpu: {}, mem: {}", cpu, mem);
  println!("order: {:?}", sequence);

  println!("===========   Completed   =========");
}
