use std::io::Read;  // read_to_string

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
  fn new(tensors: Vec<usize>) -> TensorNetwork {
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

fn bruteforce_contraction(tensors: Vec<usize>) -> (Vec<usize>,Dimension,Dimension) {
  let xor = tensors.iter().fold(0, |xor, t| xor^t);
  let n_tn = 1<<(xor.count_zeros() - xor.leading_zeros());  // 2^number of closed legs
  let mut indices_by_popcount:Vec<usize> = (0..n_tn).collect();
  indices_by_popcount.sort_by_key(|i| i.count_ones());
  let mut tn_vec = vec![TensorNetwork { cpu:Dimension::max_value(), mem:0, contracted:0, parent:0, tensors: Vec::new() };n_tn];
  tn_vec[0] = TensorNetwork::new(tensors);
  for &i in indices_by_popcount.iter() {
    for child in tn_vec[i].generate_children() {
      if child.cpu < tn_vec[child.contracted].cpu {
        tn_vec[child.contracted] = child.clone();   // need to clone tensors
      }
    }
  }

  let mut sequence = Vec::new();
  let mut i = n_tn-1;
  while i != 0 {
    sequence.push(tn_vec[i].contracted);
    i = tn_vec[i].parent;
  }
  sequence.reverse();
  (sequence, tn_vec[n_tn-1].cpu, tn_vec[n_tn-1].mem)
}

fn tensors_from_input(filename:&String) -> Vec<usize> {
  let mut file = match std::fs::File::open(filename) {
    Ok(file) => file,
    Err(_) => panic!("Cannot open input file"),
  };
  let mut contents = String::new();
  match file.read_to_string(&mut contents) {
    Ok(contents) => contents,
    Err(_) => panic!("Failed to read input"),
  };
  println!("{}",contents);
  let tensors = vec![3,21,74,172];
  tensors
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
  let args: Vec<_> = std::env::args().collect();
  if args.len() < 2 {
    panic!("No input file given");
  }
  println!("take input from file: {}", args[1]);
  let tensors = tensors_from_input(&args[1]);

  let (sequence,cpu,mem) = bruteforce_contraction(tensors);
  println!("contraction sequence: {:?}", sequence);
  println!("cpu: {}, mem: {}", cpu, mem);

  println!("===========   Completed   =========");
}
