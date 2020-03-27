type Dimension = u64;

#[derive(Debug, Clone, serde::Deserialize)]
struct AbstractTensor {
  name: String,
  shape: Vec<Dimension>,
  legs: Vec<i8>,
}

#[derive(Debug, Clone)]
struct TensorNetwork<'a> {
  legs_dim: &'a Vec<Dimension>,
  cpu: Dimension,
  mem: Dimension,
  contracted: usize,
  parent: usize,
  tensors: Vec<usize>
}


impl<'a> TensorNetwork<'a> {
  fn new(legs_dim: &'a Vec<Dimension>, tensors: Vec<usize>)  -> TensorNetwork<'a> {
    let mut tn = TensorNetwork {
      legs_dim,
      cpu: 0,
      mem: 0,
      contracted: 0,
      parent: 0,
      tensors,
    };
    tn.mem = tn.tensors.iter().map(|&t| tn.measure(t)).sum();
    tn
  }

  fn measure(&self, tensor: usize) -> Dimension {
    let mut s = 1;
    for (i, d) in self.legs_dim.iter().enumerate() {
      if (tensor >> i)%2 != 0 {
        s *= d;
      }
    }
    s
  }

  fn generate_children(&self) -> Vec<TensorNetwork<'a>> {
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
          let mem = self.tensors.iter().map(|&t| self.measure(t)).sum::<Dimension>() + self.measure(*ti) + self.measure(*tj) + self.measure(ti_dot_tj);
          let child = TensorNetwork {
            cpu: self.cpu + self.measure(ti|tj),
            mem: std::cmp::max(self.mem,mem),
            contracted: self.contracted | legs,
            parent: self.contracted,
            tensors: child_tensors,
            legs_dim: self.legs_dim
          };
          children.push(child);
        }
      }
    }
  children
  }
}

/// Take tensors represented as usize integers, with bit i=1 => tensor has leg i.
/// Legs must be sorted: first every legs to contract in the TN, then open legs.
fn bruteforce_contraction(legs_dim: &Vec<Dimension>, tensors: Vec<usize>)  -> (Vec<usize>,Dimension,Dimension) {

  // initialize suff
  let xor = tensors.iter().fold(0, |xor, t| xor^t);
  let n_tn = 1<<(xor.count_zeros() - xor.leading_zeros());  // 2^number of closed legs
  let mut indices_by_popcount:Vec<usize> = (0..n_tn).collect();
  indices_by_popcount.sort_by_key(|i| i.count_ones());
  let mut tn_vec = vec![TensorNetwork{cpu:Dimension::max_value(),mem:0,contracted:0,parent:0,tensors: Vec::new(),legs_dim}; n_tn];
  tn_vec[0] = TensorNetwork::new(legs_dim,tensors);

  // ==> Core of the programm here <==
  for &i in indices_by_popcount.iter() {
    for child in tn_vec[i].generate_children() {
      if child.cpu < tn_vec[child.contracted].cpu {
        tn_vec[child.contracted] = child.clone();   // need to clone tensors
      }
    }
  }

  // return readable result
  let mut sequence = Vec::new();
  let mut i = n_tn-1;
  while i != 0 {
    sequence.push(tn_vec[i].contracted);
    i = tn_vec[i].parent;
  }
  sequence.reverse();
  (sequence, tn_vec[n_tn-1].cpu, tn_vec[n_tn-1].mem)
}

fn tensors_from_input(input:&String) -> Vec<AbstractTensor> {
  let file = std::fs::File::open(input).expect("Cannot open input file");
  let reader = std::io::BufReader::new(file);
  let tensors: Vec<AbstractTensor> = serde_json::from_reader(reader).expect("JSON was not well-formatted");
  tensors
}

fn usize_tensor_repr(tensors: &Vec<AbstractTensor>) -> (Vec<Dimension>, Vec<usize>) {
  let mut legs_map = std::collections::HashMap::new();
  for t in tensors {
    for (i,&l) in t.legs.iter().enumerate() {
      if t.legs.iter().filter(|&&l2| l2 == l).count() > 1 {
        panic!("A vector has twice the same leg. Trace is not allowed.");
      }

      let v = legs_map.insert(l, (true,t.shape[i]));
      if v != None {
        let (once,shape) = v.unwrap();
        if !once { panic!("A given leg appears more than twice"); }
        if shape != t.shape[i] { panic!("A given leg has two diffent dimensions"); }
        legs_map.insert(l, (false,t.shape[i]));
      }
    }
  }

  let mut legs_indices: Vec<i8> = legs_map.keys().map(|&l| l).collect();
  legs_indices.sort_by_key(|l| (legs_map[l].0, l.abs()));
  let legs_dim: Vec<Dimension> = legs_indices.iter().map(|l| legs_map[l].1).collect();

  let mut repr = vec![0;tensors.len()];
  for (i,t) in tensors.iter().enumerate() {
    for &lt in t.legs.iter() {
      repr[i] |= 1 << legs_indices.iter().position(|&l| lt==l).unwrap();
    }
  }
  (legs_dim,repr)
}

fn main() {
  println!("=============   Begin   ===========");

  let args: Vec<_> = std::env::args().collect();
  let input = if args.len() > 2 {
    println!("take input from file: {}", args[1]);
    args[1].clone()
  } else {
    println!("No input file given, call input_sample.json");
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
    String::from("input_sample.json")
  };
  let tensors = tensors_from_input(&input);
  let (legs_dim, tensor_repr) = usize_tensor_repr(&tensors);

  let (sequence,cpu,mem) = bruteforce_contraction(&legs_dim, tensor_repr);
  println!("contraction sequence: {:?}", sequence);
  println!("cpu: {}, mem: {}", cpu, mem);

  println!("===========   Completed   =========");
}
