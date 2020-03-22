#[derive(Debug)]
struct AbstractTensor {
  name: String,
  legs: Vec<i8>,
  shape: Vec<u64>,
  size: u64,
  ndim: u8,
}

impl AbstractTensor {
  pub fn new(name: String,  legs: Vec<i8>, shape: Vec<u64>) -> AbstractTensor {
    AbstractTensor {
      size: shape.iter().product(),
      ndim: legs.len() as u8,
      name,
      legs,
      shape,
    }
  }
}

#[derive(Debug)]
struct TensorNetwork {
  cpu: u64,
  mem: u64,
  contracted: u64,
  n_tens: u8,
  tensors: Vec<AbstractTensor>,
}


impl TensorNetwork {
  pub fn new(tensors: Vec<AbstractTensor>) -> TensorNetwork {
    let mut mem: u64 = 0;
    for t in &tensors {
      mem += t.size;
    }
    TensorNetwork {
      cpu: 0,
      mem,
      contracted: 0,
      n_tens: tensors.len() as u8,
      tensors,
    }
  }
}


fn main() {
  println!("=============   Begin   ===========");

  let t = AbstractTensor::new(String::from("T"),vec!(0,1,2),vec!(10,10,9));
  let c = AbstractTensor::new(String::from("C"),vec!(0,3),vec!(10,10));
  println!("{:?}\n{:?}",t,c);
  let tn = TensorNetwork::new(vec!(t,c));
  println!("{:?}",tn);

  println!("===========   Completed   =========");
}
