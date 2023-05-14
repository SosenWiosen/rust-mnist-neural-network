use crate::data::VectorizedLabelPixelsPair;
use crate::math_helpers::{sigmoid, sigmoid_prime};
use ndarray::{Array, Array1, Array2, Ix};
use ndarray_rand::rand::seq::SliceRandom;
use ndarray_rand::rand::thread_rng;
use ndarray_rand::rand_distr::num_traits::ToPrimitive;
use ndarray_rand::rand_distr::Normal;
use ndarray_rand::RandomExt;
use std::iter::zip;

#[derive(Debug)]
pub(crate) struct Network {
    number_of_layers: usize,
    sizes: Vec<usize>,
    biases: Vec<Array1<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub(crate) fn new(sizes: Vec<usize>) -> Network {
        let number_of_layers = sizes.len() as usize;
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
        &mut self,
        mut training_data: Vec<VectorizedLabelPixelsPair>,
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
    fn update_mini_batch(&mut self, mini_batch: &[VectorizedLabelPixelsPair], learning_rate: f64) {
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
            nabla_b = zip(nabla_b, delta_nabla_b)
                .map(|(nb, dnb)| nb + dnb)
                .collect();
            nabla_w = zip(nabla_w, delta_nabla_w)
                .map(|(nw, dnw)| nw + dnw)
                .collect();
        }
        self.weights = zip(self.weights.clone(), nabla_w)
            .map(|(w, nw)| w - (learning_rate / nw.len() as f64) * nw)
            .collect();
        self.biases = zip(self.biases.clone(), nabla_b)
            .map(|(w, nb)| w - (learning_rate / nb.len() as f64) * nb)
            .collect();
    }
    fn backpropagation(
        &self,
        vector_label_pixels_pair: &VectorizedLabelPixelsPair,
    ) -> (Vec<Array1<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array1<f64>> = Vec::new();
        for bias in self.biases.iter() {
            nabla_b.push(Array1::zeros(bias.raw_dim()))
        }
        let mut nabla_w: Vec<Array2<f64>> = Vec::new();
        for weight in self.weights.iter() {
            nabla_w.push(Array2::zeros(weight.raw_dim()))
        }

        //feedforward
        let activation = vector_label_pixels_pair.0.clone();
        let mut activations: Vec<Array1<f64>> = vec![activation];
        let mut z_vectors: Vec<Array1<f64>> = Vec::new();
        for (b, w) in zip(self.biases.iter(), self.weights.iter()) {
            let z = w.dot(activations.last().unwrap()) + b;
            z_vectors.push(z.clone());
            let activation = z.iter().map(|x| sigmoid(*x)).collect();
            activations.push(activation);
        }

        //backward pass
        let delta = self.cost_derivative(activations.last().unwrap(), &vector_label_pixels_pair.1)
            * z_vectors.last().unwrap().map(|x| sigmoid_prime(*x));

        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1] = delta.clone();

        let nabla_w_len = nabla_w.len();

        let x = activations[activations.len() - 2].clone().reversed_axes();
        nabla_w[nabla_w_len - 1] = delta.dot(&x);
        for l in 2usize..self.number_of_layers {
            let z = z_vectors[z_vectors.len() - l].clone();
            let sp = z.map(|x| sigmoid_prime(*x));
            let delta = self.weights[self.weights.len() - l + 1]
                .clone()
                .reversed_axes()
                .dot(&delta)
                * sp;
            let nabla_b_len = nabla_b.len();
            nabla_b[nabla_b_len - l] = delta.clone();
            let nabla_w_len = nabla_w.len();
            nabla_w[nabla_w_len - l] = delta.dot(&activations[activations.len() - l - 1].clone());
        }
        todo!("finish backpropagation");
    }
    fn cost_derivative(&self, output_activations: &Array1<f64>, y: &Array1<f64>) -> Array1<f64> {
        output_activations - y
    }
}
