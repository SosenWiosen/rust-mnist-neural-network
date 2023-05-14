mod data;
mod math_helpers;
mod network;

use crate::data::get_data;
use ndarray::{Array, Array1, Array2, Ix};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use network::Network;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();

    let filepath = &args[1];
    let (train, test) = match get_data(filepath) {
        Ok(data) => data,
        Err(_) => panic!("Problem loading the data!"),
    };
    println!("Finished loading data, training data size: {}", train.len());
    let mut net = Network::new(vec![784, 30, 10]);
    net.stochastic_gradient_descent(train, test, 30, 10, 3.0);
}
