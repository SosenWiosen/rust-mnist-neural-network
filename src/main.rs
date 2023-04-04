mod math_helpers;
mod network;

use ndarray::{Array, Array1, Array2, Ix};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use network::Network;

fn main() {
    println!("{:?}", Network::new(vec![2, 3, 2]));
}
