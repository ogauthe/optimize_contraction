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
      /*else {
        if tensor >> i == 0 {
          break;
        }
      }*/
    }
    /*let mut t = tensor;
    let mut i = 0;
    while t != 0 {
      if t%2 { s *= self.legs_dim[i]; }
      i += 1
      t = t >> 1;
    }*/
    s
  }

  fn generate_children(&self) -> Vec<TensorNetwork<'a>> {
    let mut children = Vec::new();
    for (i,&ti) in self.tensors.iter().enumerate() {
      for (j,&tj) in self.tensors[i+1..].iter().enumerate() {
        let legs = ti & tj;
        if legs != 0 {  // if tensors have common leg
          let mut child_tensors = self.tensors.clone();
          child_tensors.remove(i+1+j);
          child_tensors.remove(i);
          let ti_dot_tj = ti^tj;
          child_tensors.push(ti_dot_tj);
          let mem = self.tensors.iter().map(|&t| self.measure(t)).sum::<Dimension>() + self.measure(ti) + self.measure(tj) + self.measure(ti_dot_tj);
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


fn greedy_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<usize>)  -> (Vec<usize>,Dimension,Dimension) {
  let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
  let max_tn = (1<<(xor.count_zeros() - xor.leading_zeros())) - 1;  // 2^number of closed legs - 1
  let mut tn_vec = vec![TensorNetwork::new(legs_dim,tensor_repr)];
  let mut last_contracted = 0;
  while last_contracted < max_tn {
    let children = tn_vec.last().unwrap().generate_children();
    let greedy_child = children.iter().min_by_key(|&c| c.cpu).unwrap();
    tn_vec.push(greedy_child.clone());
    last_contracted = greedy_child.contracted;
  }
  let final_tn = tn_vec.last().unwrap();
  (tn_vec.iter().map(|tn| tn.contracted).collect(),final_tn.cpu,final_tn.mem)
}

/// Take tensors represented as usize integers, with bit i=1 => tensor has leg i.
/// Legs must be sorted: first every legs to contract in the TN, then open legs.
fn exhaustive_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<usize>)  -> (Vec<usize>,Dimension,Dimension) {

  // initialize suff
  let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
  let max_tn = (1<<(xor.count_zeros() - xor.leading_zeros())) - 1;  // 2^number of closed legs - 1
  let mut indices_by_popcount:Vec<usize> = (0..max_tn+1).collect();
  indices_by_popcount.sort_by_key(|i| i.count_ones());
  let greedy = greedy_search(legs_dim, tensor_repr.clone()); // no need to continue paths that go beyond any final result
  println!("greedy result: {:?}",greedy);
  let mut tn_vec = vec![TensorNetwork{cpu:greedy.1,mem:greedy.2,contracted:0,parent:0,tensors: Vec::new(),legs_dim}; max_tn+1];
  tn_vec[0] = TensorNetwork::new(legs_dim,tensor_repr);
  tn_vec[max_tn].parent = greedy.0[greedy.0.len()-2];  // in case greedy search is optimal

  // ==> Core of the programm here <==
  let mut count = 0;
  for &i in indices_by_popcount.iter() {
    for child in tn_vec[i].generate_children() {
      if child.cpu < std::cmp::min(tn_vec[child.contracted].cpu,tn_vec[max_tn].cpu) {
        count += 1;
        tn_vec[child.contracted] = child.clone();   // need to clone tensors
      }
    }
  }
  println!("number of computed tn: {}",tn_vec.iter().filter(|tn| tn.cpu<greedy.1).count());
  println!("count: {}",count);

  // return representation of contracted leg sequence as result
  let mut sequence_repr = Vec::new();
  let mut i = max_tn;
  while i != 0 {
    sequence_repr.push(tn_vec[i].contracted);
    i = tn_vec[i].parent;
  }
  sequence_repr.push(0);
  sequence_repr.reverse();
  (sequence_repr, tn_vec[max_tn].cpu, tn_vec[max_tn].mem)
}

fn tensors_from_input(input:&String) -> Vec<AbstractTensor> {
  let file = std::fs::File::open(input).expect("Cannot open input file");
  let reader = std::io::BufReader::new(file);
  let tensors: Vec<AbstractTensor> = serde_json::from_reader(reader).expect("JSON was not well-formatted");
  tensors
}

fn represent_usize(tensors: &Vec<AbstractTensor>) -> (Vec<i8>, Vec<Dimension>, Vec<usize>) {
  let mut legs_map = std::collections::HashMap::new();
  for t in tensors {
    for (i,&l) in t.legs.iter().enumerate() {
      if t.legs.iter().filter(|&&l2| l2 == l).count() > 1 {
        panic!("A vector has twice the same leg. Trace is not allowed.");
      }
      if let Some((once,dim)) = legs_map.insert(l, (true,t.shape[i])) {
        if !once { panic!("Leg {} appears more than twice",l); }
        if dim != t.shape[i] { panic!("Leg {} has two diffent dimensions",l); }
        legs_map.insert(l, (false,t.shape[i]));
      }
    }
  }
  if legs_map.keys().len() > 64 {
    panic!("Cannot consider more than 64 legs");
  }
  let mut legs_indices: Vec<i8> = legs_map.keys().map(|&l| l).collect();
  legs_indices.sort_by_key(|l| (legs_map[l].0, l.abs()));
  let legs_dim: Vec<Dimension> = legs_indices.iter().map(|l| legs_map[l].1).collect();

  let mut tensor_repr = vec![0;tensors.len()];
  for (i,t) in tensors.iter().enumerate() {
    for &lt in t.legs.iter() {
      tensor_repr[i] |= 1 << legs_indices.iter().position(|&l| lt==l).unwrap();
    }
  }
  (legs_indices,legs_dim,tensor_repr)
}

fn sequence_from_repr(legs_indices: &Vec<i8>, sequence_repr: Vec<usize>) -> Vec<Vec<i8>> {
  let mut sequence = Vec::new();
  for i in 1..sequence_repr.len() {
    let mut legs = Vec::new();
    let mut legs_repr = sequence_repr[i-1]^sequence_repr[i];
    let mut j = 0;
    while legs_repr != 0 {
      if legs_repr%2 !=0 {
        legs.push(legs_indices[j]);
      }
      legs_repr = legs_repr >> 1;
      j += 1;
    }
    sequence.push(legs);
  }
  sequence
}



fn main() {
  println!("Find optimal contraction sequence of a tensor network.");

  let args: Vec<_> = std::env::args().collect();
  let input = if args.len() > 1 {
    println!("Take input from file: {}", args[1]);
    args[1].clone()
  } else {
    println!("No input file given, take input from input_sample.json");
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
     * Sequence repr: 0 -> (1 or 2) -> 3 -> 15
     * Sequence: (0,), (1,), (2,3)
     */
    String::from("input_sample.json")
  };
  let tensors = tensors_from_input(&input);
  println!("Tensors:");
  for t in &tensors {
    println!("{:?}",t);
  }
  let (legs_indices, legs_dim, tensor_repr) = represent_usize(&tensors);

  println!("\nLaunch bruteforce search for best contraction sequence...");
  let (sequence_repr,cpu,mem) = exhaustive_search(&legs_dim, tensor_repr);
  println!("Done. cpu: {}, mem: {}", cpu, mem);
  let sequence = sequence_from_repr(&legs_indices, sequence_repr);
  println!("Sequence: {:?}",sequence);

}
