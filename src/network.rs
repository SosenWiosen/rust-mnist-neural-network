use crate::math_helpers::sigmoid;
use ndarray::{Array, Array1, Array2, Ix};
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;

#[derive(Debug)]
pub(crate) struct Network {
    number_of_layers: u32,
    sizes: Vec<u32>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub(crate) fn new(sizes: Vec<u32>) -> Network {
        let number_of_layers = sizes.len() as u32;
        let distribution = Normal::new(0., 1.0).unwrap();

        let biases = sizes[1..]
            .iter()
            .map(|&x| Array::random(x as Ix, distribution))
            .collect();
        let weights = sizes[..sizes.len()]
            .iter()
            .zip(sizes[1..].iter())
            .map(|(&x, &y)| Array::random((y as Ix, x as Ix), distribution))
            .collect();
        Network {
            number_of_layers,
            sizes,
            biases,
            weights,
        }
    }

    fn feedforward(&self, input: Array1<f64>) -> Array1<f64> {
        let mut activation = input;
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            activation = w
                .dot(&activation)
                .iter()
                .zip(b.iter())
                .map(|(x, y)| sigmoid(x + y))
                .collect()
        }
        activation
    }
}
