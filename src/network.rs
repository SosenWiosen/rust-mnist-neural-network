use crate::data::VectorLabelPixelsPair;
use crate::math_helpers::sigmoid;
use ndarray::{Array, Array1, Array2, Ix};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
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

    fn feed_forward(&self, input: Array1<f64>) -> Array1<f64> {
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
    fn stochastic_gradient_descent(
        &self,
        mut training_data: Vec<VectorLabelPixelsPair>,
        epochs: u16,
        mini_batch_size: u16,
        eta: f64,
    ) {
        let training_data_len = training_data.len();
        for i in 0..epochs {
            training_data.shuffle(&mut thread_rng());
            let mut mini_batches = training_data.chunks(mini_batch_size.to_usize().unwrap());
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }
            println!("Epoch {i} complete");
        }
    }
    fn update_mini_batch(&self, mini_batch: &[VectorLabelPixelsPair], learning_rate: f64) {
        let mut nabla_b: Vec<Array1<f64>> = Vec::new();
        for bias in self.biases.iter() {
            nabla_b.push(Array1::zeros(bias.raw_dim()))
        }
        let mut nabla_w: Vec<Array2<f64>> = Vec::new();
        for weight in self.weights.iter() {
            nabla_w.push(Array2::zeros(weight.raw_dim()))
        }
        for vector_label_pixels_pair in mini_batch {
            let (delta_nabla_b, delta_nabla_w) = self.backpropagation(vector_label_pixels_pair);
        }
        todo!("Implement rest of logic needed for updating minibatches ")
    }
    fn backpropagation(
        &self,
        vector_label_pixels_pair: &VectorLabelPixelsPair,
    ) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        // let mut nabla_b: Vec<Array1<f64>> = Vec::new();
        // for bias in self.biases.iter() {
        //     nabla_b.push(Array1::zeros(bias.raw_dim()))
        // }
        // let mut nabla_w: Vec<Array2<f64>> = Vec::new();
        // for weight in self.weights.iter() {
        //     nabla_w.push(Array2::zeros(weight.raw_dim()))
        // }
        todo!("Implement backpropagation")
    }
}
