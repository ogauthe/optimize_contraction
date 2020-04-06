use std::time::Instant;
use std::collections::hash_map::Entry;
use fnv::FnvHashMap;  // faster than std for integer keys

type Dimension = u64;
type BinaryTensor = u64;

#[derive(Debug, Clone, serde::Deserialize)]
struct AbstractTensor { // convenient struct to read json input file
  name: String,            // not currently used
  shape: Vec<Dimension>,
  legs: Vec<i8>,
}

// Define binary representation for tensors: bit i=1 => tensor has open leg i.
// Legs are sorted: first every legs to contract in the TN, then open legs.
// A TN is uniquely identified by the legs that have been contracted, in binary form.
#[derive(Debug, Clone)]
struct TensorNetwork<'a> {
  legs_dim: &'a Vec<Dimension>,   // define a binary representation for TN legs
  cpu: Dimension,     // cpu cost to reach this TN (sum of all past steps)
  mem: Dimension,     // upper bound for needed memory to reach this TN (assume copies at each contraction)
  id: BinaryTensor,           // binary representation of contracted legs: bit i is 1 if leg i has been contracted
  parent: BinaryTensor,       // id of parent TN, initial TN (id 0) is its own parent
  tensors: Vec<BinaryTensor>, // binary representation of tensors in TN
  allows_outer: Vec<bool>     // whether tensor i construction allows outer product
}


impl<'a> TensorNetwork<'a> {
  fn new(legs_dim: &'a Vec<Dimension>, tensors: Vec<BinaryTensor>)  -> TensorNetwork<'a> {
    let mut tn = TensorNetwork {
      legs_dim,
      cpu: 0,
      mem: 0,
      id: 0,
      parent: 0,
      allows_outer: vec![true; tensors.len()],
      tensors,
    };
    tn.mem = tn.tensors.iter().map(|&t| tn.checked_measure(t).unwrap()).sum();
    tn
  }

  fn measure(&self, tensor: BinaryTensor) -> Dimension {
    let mut s:Dimension = 1;
    let mut t = tensor;
    for &d in self.legs_dim {
      if t%2 != 0 {
        s *= d;
      } else {
        if t == 0 {
          return s;
        }
      }
      t >>= 1;
    }
    s
  }

  fn checked_measure(&self, tensor: BinaryTensor) -> Option<Dimension> {  // deal with high risk of overflow (other solution: Dimension -> f64)
    let mut s:Dimension = 1;
    let mut t = tensor;
    for &d in self.legs_dim {
      if t%2 != 0 {
        s = s.checked_mul(d)?;
      } else {
        if t == 0 {
          return Some(s);
        }
      }
      t >>= 1;
    }
    Some(s)
  }

  fn contract_tensors(&self, i0:usize, j0:usize) -> Option<TensorNetwork<'a>> { // compute TN child obtained by contraction of ti and tj
    let i = std::cmp::min(i0,j0);
    let j = std::cmp::max(i0,j0);
    let ti = self.tensors[i];                                                 // of tensors i and j. Returns None if no common
    let tj = self.tensors[j];                                                 // leg or cpu overflow.
    let legs = ti & tj;
    if legs == 0 { return None; }  // do not consider outer product
    let cpu = self.cpu.checked_add(self.checked_measure(ti|tj)?)?;  // test overflow the soonest possible
    let ti_dot_tj = ti^tj;  // measure(ti^tj) < measure(ti|tj), which has been checked
    let ti_size = self.measure(ti); // tensors in self.tensors are checked at contruction
    let tj_size = self.measure(tj);
    let legs_sizep3 = self.measure(legs).saturating_pow(3);  // measure(legs) < measure(ti|tj)
    let mut child_tensors = self.tensors.clone();
    child_tensors.remove(j);
    child_tensors.remove(i);
    child_tensors.push(ti_dot_tj);
    let mut child_allow = self.allows_outer.clone();
    child_allow.remove(j);
    child_allow.remove(i);
    let allow_tij = ((ti_dot_tj & tj == 0) && legs_sizep3 > ti_size) || ((ti_dot_tj & ti == 0) && legs_sizep3 > tj_size);
    child_allow.push(allow_tij);
    // Assume copies of both ti and tj are needed in order to arange legs for BLAS (upper bound)
    // Assume ti and tj are destroyed after copy, and their copies are destroyed after contraction
    // then max memory is sum(t_mem, including i and j) + max_mem(ti,tj,ti.tj)
    let mem = self.tensors.iter().map(|&t| self.measure(t)).sum::<Dimension>() + std::cmp::max(ti_size,std::cmp::max(tj_size,self.measure(ti_dot_tj)));  // cpu cost is always > memory cost, hence no overflow here
    Some(TensorNetwork {
      legs_dim: self.legs_dim,
      cpu,
      mem: std::cmp::max(self.mem,mem), // mem is an upper bond for whole contraction.
      parent: self.id,
      id: self.id | legs,
      tensors: child_tensors,
      allows_outer: child_allow,
    })
  }


  fn generate_children(&self) -> Vec<TensorNetwork<'a>> {
    for (k,&tk) in self.tensors.iter().enumerate().filter(|&(k,_)| self.allows_outer[k]) {
      if self.tensors.iter().enumerate().filter(|&(i,ti)| i != k && (ti^tk)&ti == 0).count() > 1 {
        println!("Warning: outer product (not implemented) may be necessary to contract TN {}",self.id)
      }
    }
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


fn greedy_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<BinaryTensor>)  -> (Vec<BinaryTensor>,TensorNetwork) {
  let max_tn = {
    let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
    (1<<(xor.count_zeros() - xor.leading_zeros())) - 1  // 2^number of closed legs - 1
  };
  let mut tn = TensorNetwork::new(legs_dim,tensor_repr);
  tn.allows_outer = vec![false;tn.tensors.len()];  // do not consider outer product in greedy (cannot become true later)
  let mut sequence_repr = vec![0];
  while tn.id < max_tn {
    tn = tn.generate_children().iter().min_by_key(|&c| c.cpu).unwrap().clone();
    sequence_repr.push(tn.id);
  }
  (sequence_repr,tn)
}

fn exhaustive_search(legs_dim: &Vec<Dimension>, tensor_repr: Vec<BinaryTensor>)  -> (Vec<BinaryTensor>,TensorNetwork) {

  // first execute greedy search as reasonable upper bound.
  let mut best = greedy_search(legs_dim, tensor_repr.clone()).1;
  println!("\ngreedy result: {:?}", best);

  // initialize suff
  let n_c = {
    let xor = tensor_repr.iter().fold(0, |xor, t| xor^t);
    (xor.count_zeros() - xor.leading_zeros()) as usize
  };
  let mut generation_maps = vec![FnvHashMap::default(); n_c];  // put fully contracted outside map (access without hash cost)
  generation_maps[0].insert(0,TensorNetwork::new(legs_dim,tensor_repr));

  // ==> Core of the programm here <==
  println!("\nLaunch exhaustive search for best contraction sequence...");
  let start = Instant::now();
  for generation in 0..n_c {
    let (current_generation, next_generations) = generation_maps[generation..].split_first_mut().unwrap();
    for parent in current_generation.values() {
      if parent.cpu < best.cpu {    // Do not explore path already more expensive than currrent best result.
        for child in parent.generate_children() { // best.cpu may change, cannot filter iter
          if child.cpu < best.cpu {  // bad children loose their turn
            let child_generation = child.id.count_ones() as usize;
            if child_generation == n_c {
              best = child.clone();
            } else {
              match next_generations[child_generation-generation-1].entry(child.id) { // evalutate hash function only once
                Entry::Vacant(entry) => { entry.insert(child.clone()); },
                Entry::Occupied(mut entry) => if child.cpu < entry.get().cpu {
                  entry.insert(child.clone());
                }    // do nothing if current entry is better than child
              }
            }
          }
        }
      }
    }
  }
  let duration = start.elapsed();
  println!("Done. Time elapsed = {:?}", duration);
  // ==> end of expensive part <==

  // return representation of contracted leg sequence as result
  let mut sequence_repr = vec![best.id, best.parent];
  let mut id = best.parent;
  while id != 0 {
    id = generation_maps[id.count_ones() as usize].get(&id).unwrap().parent;
    sequence_repr.push(id);
  }
  sequence_repr.reverse();
  (sequence_repr, best)
}

fn tensors_from_input(input: &str) -> Vec<AbstractTensor> {
  let file = std::fs::File::open(input).expect("Cannot open input file");
  let reader = std::io::BufReader::new(file);
  let tensors: Vec<AbstractTensor> = serde_json::from_reader(reader).expect("JSON was not well-formatted");
  tensors
}

fn represent_binary(tensors: &Vec<AbstractTensor>) -> (Vec<i8>, Vec<Dimension>, Vec<BinaryTensor>) {
  let mut legs_map = FnvHashMap::default();
  for t in tensors {
    if t.legs.len() != t.shape.len() {
      panic!("Tensor {} shape and dimension have different lengths",t.name);
    }
    for (i,&l) in t.legs.iter().enumerate() {
      if t.shape[i] < 2 {
        panic!("Tensor {} has leg {} with forbidden dimension {}.",t.name,i,t.shape[i]);
      }
      if t.legs.iter().filter(|&&l2| l2 == l).count() > 1 {
        panic!("Tensor {} has twice the same leg. Trace is not allowed.", t.name);
      }
      match legs_map.entry(l) {
        Entry::Vacant(entry) => { entry.insert((true,t.shape[i])); }
        Entry::Occupied(mut entry) => {
          let (ref mut once, ref dim) = entry.get_mut();
          if !*once { panic!("Leg {} appears more than twice",l); }
          if *dim != t.shape[i] { panic!("Leg {} has two diffent dimensions",l); }
          *once = false;
        }
      }
    }
  }
  if legs_map.keys().len() > (0 as BinaryTensor).count_zeros() as usize {
    panic!("Cannot consider more than {} legs",(0 as BinaryTensor).count_zeros());
  }

  let mut legs_indices: Vec<i8> = legs_map.keys().map(|&l| l).collect();
  // expect l<0 for free legs and l>0 for legs to contract (not mandatory)
  legs_indices.sort_by_key(|l| (legs_map[l].0, l.abs()));
  let legs_dim: Vec<Dimension> = legs_indices.iter().map(|l| legs_map[l].1).collect();

  let mut tensor_repr = vec![0;tensors.len()];
  for (i,t) in tensors.iter().enumerate() {
    for &lt in t.legs.iter() {
      tensor_repr[i] |= 1 << legs_indices.iter().position(|&l| lt==l).unwrap();
    }
  }

  if !is_connex(&tensor_repr) {
    panic!("Tensor network is not connex.");
  }

  (legs_indices,legs_dim,tensor_repr)
}

fn is_connex(tensor_repr: &Vec<BinaryTensor>) -> bool {
  let mut to_reach = tensor_repr.clone();
  let mut to_visit = vec![to_reach.pop().unwrap()];
  while let Some(t) = to_visit.pop() {
    for i in (0..to_reach.len()).rev() {
      if t & to_reach[i] != 0 {
        to_visit.push(to_reach.remove(i));
        if to_reach.is_empty() {
          return true;
        }
      }
    }
  }
  false
}

fn sequence_from_repr(legs_indices: &Vec<i8>, sequence_repr: Vec<BinaryTensor>) -> Vec<Vec<i8>> {
  let mut sequence = Vec::new();
  for i in 1..sequence_repr.len() {
    let mut legs = Vec::new();
    let mut legs_repr = sequence_repr[i-1]^sequence_repr[i];
    let mut j = 0;
    while legs_repr != 0 {
      if legs_repr%2 !=0 {
        legs.push(legs_indices[j]);
      }
      legs_repr >>= 1;
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
    &args[1]
  } else {
    println!("No input file given, take input from input_sample.json");
    /* input sample: symmetric CTMRG step.
     *
     *   C-0-T- -1
     *   |   |
     *   1   2
     *   |   |
     *   T-3-E- -2
     *   |   |
     *  -3   -4
     *
     * tensor binary representations:
     *  C: 3
     *  T1: 21
     *  T2: 74
     *  E: 172
     *
     * TN sequence binary representation: 0 -> (1 or 2) -> 3 -> 15
     * leg contraction sequence: (0,), (1,), (2,3)
     */
    "input_sample.json"
  };
  let tensors = tensors_from_input(&input);
  println!("Tensors:");
  for t in &tensors {
    println!("{:?}",t);
  }
  let (legs_indices, legs_dim, tensor_repr) = represent_binary(&tensors);


  let (sequence_repr,best_tn) = exhaustive_search(&legs_dim, tensor_repr);
  let sequence = sequence_from_repr(&legs_indices, sequence_repr);
  println!("cpu: {}, mem: {}", best_tn.cpu, best_tn.mem);
  println!("leg contraction sequence: {:?}",sequence);

}
