use std::time::Instant;
use std::collections::{HashMap,hash_map::Entry};

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
  id: usize,
  parent: usize,
  tensors: Vec<usize>
}


impl<'a> TensorNetwork<'a> {
  fn new(legs_dim: &'a Vec<Dimension>, tensors: Vec<usize>)  -> TensorNetwork<'a> {
    let mut tn = TensorNetwork {
      legs_dim,
      cpu: 0,
      mem: 0,
      id: 0,
      parent: 0,
      tensors,
    };
    tn.mem = tn.tensors.iter().map(|&t| tn.checked_measure(t).unwrap()).sum();
    tn
  }

  fn measure(&self, tensor: usize) -> Dimension {
    let mut s:Dimension = 1;
    for (i, &d) in self.legs_dim.iter().enumerate() {
      if (tensor >> i)%2 != 0 {
        s *= d;
      }
    }
    s
  }

  fn checked_measure(&self, tensor: usize) -> Option<Dimension> {  // deal with high risk of overflow
    let mut s:Dimension = 1;                               // other solution: Dimension -> f64
    for (i, &d) in self.legs_dim.iter().enumerate() {
      if (tensor >> i)%2 != 0 {
        s = s.checked_mul(d)?;
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
      t >>= 1;
    }*/
    Some(s)
  }

  fn contract_tensors(&self, i:usize, j:usize) -> Option<TensorNetwork<'a>> {
    let ti = self.tensors[i];
    let tj = self.tensors[j];
    let legs = ti & tj;
    if legs == 0 { return None; }
    let cpu = self.cpu.checked_add(self.checked_measure(ti|tj)?)?;  // test overflow the soonest possible
    let mut child_tensors = self.tensors.clone();
    child_tensors.remove(std::cmp::max(i,j));
    child_tensors.remove(std::cmp::min(i,j));
    let ti_dot_tj = ti^tj;
    child_tensors.push(ti_dot_tj);
    // Assume copies of both ti and tj are needed in order to arange legs for BLAS (upper bound)
    // Assume ti and tj are destroyed after copy, and their copies are destroyed after contraction
    // then max memory is sum(t_mem, including i and j) + max_mem(ti,tj,ti.tj)
    let mem = self.tensors.iter().map(|&t| self.measure(t)).sum::<Dimension>() + std::cmp::max(self.measure(ti),std::cmp::max(self.measure(tj),self.measure(ti_dot_tj)));  // cpu cost is always > memory cost, hence no overflow here
    Some(TensorNetwork {
      legs_dim: self.legs_dim,
      cpu,
      mem: std::cmp::max(self.mem,mem), // mem is an upper bond for whole contraction.
      id: self.id | legs,
      parent: self.id,
      tensors: child_tensors,
    })
  }

  fn generate_children(&self) -> Vec<TensorNetwork<'a>> {
    let mut children = Vec::new();
    for i in 0..self.tensors.len() {
      for j in i+1..self.tensors.len() {
        if let Some(child) = self.contract_tensors(i,j) {
          children.push(child);
        }
      }
    }
  children
  }
}


fn greedy_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<usize>)  -> (Vec<usize>,TensorNetwork) {
  let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
  let max_tn = (1<<(xor.count_zeros() - xor.leading_zeros())) - 1;  // 2^number of closed legs - 1
  let mut tn = TensorNetwork::new(legs_dim,tensor_repr);
  let mut sequence_repr = vec![0];
  while tn.id < max_tn {
    tn = tn.generate_children().iter().min_by_key(|&c| c.cpu).unwrap().clone();
    sequence_repr.push(tn.id);
  }
  (sequence_repr,tn)
}

/// Take tensors represented as usize integers, with bit i=1 => tensor has leg i.
/// Legs must be sorted: first every legs to contract in the TN, then open legs.
fn exhaustive_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<usize>)  -> (Vec<usize>,TensorNetwork) {

  // first execute greedy search as reasonable upper bound. Do not explore path more expensive than greedy result.
  let mut best = greedy_search(legs_dim, tensor_repr.clone()).1;
  println!("greedy result: {:?}", best);

  // initialize suff
  let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
  let n_c = (xor.count_zeros() - xor.leading_zeros()) as usize;
  let max_tn = (1<<n_c) - 1;  // 2^number of closed legs - 1
  let mut generation_maps = vec![HashMap::new(); n_c];  // put fully contracted outside map (access without hash cost)
  generation_maps[0].insert(0,TensorNetwork::new(legs_dim,tensor_repr));

  // first execute greedy search as reasonable upper bound. Do not explore path more expensive than greedy result.
  let mut best = greedy_search(legs_dim, tensor_repr.clone()).1;
  println!("greedy result: {:?}",best);

  // ==> Core of the programm here <==
  for generation in 0..n_c {
    let (current_generation, next_generations) = generation_maps[generation..].split_first_mut().unwrap();
    for parent in current_generation.values() { // best.cpu may change, cannot filter iter
      if parent.cpu < best.cpu {
        for child in parent.generate_children() {  // best.cpu may change, cannot filter iter
          if child.cpu < best.cpu {  // bad children loose their place
            let child_generation = child.id.count_ones() as usize;
            if child_generation == n_c {
              best = child.clone();
            } else {
              match next_generations[child_generation-generation-1].entry(child.id) {
                Entry::Occupied(mut entry) => if child.cpu < entry.get().cpu {
                  entry.insert(child.clone()); // keep current if better than child
                },
                Entry::Vacant(entry) => { entry.insert(child.clone()); }
              }
            }
          }
        }
      }
    }
  }

  // return representation of contracted leg sequence as result
  let mut sequence_repr = vec![max_tn,best.parent];
  let mut i = best.parent;
  while i != 0 {
    i = generation_maps[i.count_ones() as usize].get(&i).unwrap().parent;
    sequence_repr.push(i);
  }
  sequence_repr.reverse();
  (sequence_repr, best.clone())
}

fn tensors_from_input(input:&String) -> Vec<AbstractTensor> {
  let file = std::fs::File::open(input).expect("Cannot open input file");
  let reader = std::io::BufReader::new(file);
  let tensors: Vec<AbstractTensor> = serde_json::from_reader(reader).expect("JSON was not well-formatted");
  tensors
}

fn represent_usize(tensors: &Vec<AbstractTensor>) -> (Vec<i8>, Vec<Dimension>, Vec<usize>) {
  let mut legs_map = HashMap::new();
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
  let start = Instant::now();
  let (sequence_repr,best_tn) = exhaustive_search(&legs_dim, tensor_repr);
  let duration = start.elapsed();
  println!("Done. Time elapsed = {:?}", duration);
  let sequence = sequence_from_repr(&legs_indices, sequence_repr);
  println!("cpu: {}, mem: {}", best_tn.cpu, best_tn.mem);
  println!("leg contraction sequence: {:?}",sequence);

}
