#[derive(Debug)]
struct AbstractTensor {
  name: String,
  legs: Vec<i8>,
  shape: Vec<u64>,
  size: u64,
  ndim: u8,
}

fn build_abstract_tensor(name: String,  legs: Vec<i8>, shape: Vec<u64>) -> AbstractTensor {
    AbstractTensor {
    size: shape.iter().product(),
    ndim: legs.len() as u8,
    name,
    legs,
    shape,
  }
}

impl AbstractTensor {
  fn dot(&self, &t: AbstractTensor) -> AbstractTensor {
    c_legs = self.find_common_legs(t)
    legsA = 
  }
}


fn main() {
  println!("=============   Begin   ===========");

  let t = build_abstract_tensor(String::from("T"),vec!(0,1,2),vec!(10,10,9));
  println!("{:?}",t);


  println!("===========   Completed   =========");
}
