use crate::data::{LabelPixelsPair, VectorizedLabelPixelsPair};
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
    biases: Vec<Array2<f64>>,
    weights: Vec<Array2<f64>>,
}

impl Network {
    pub(crate) fn new(sizes: Vec<usize>) -> Network {
        let number_of_layers = sizes.len() as usize;
        let distribution = Normal::new(0.0, 1.0).unwrap();

        let biases: Vec<Array2<f64>> = sizes[1..]
            .iter()
            .map(|&x| Array::random((x as Ix, 1), distribution))
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

    fn feed_forward(&self, input: Array2<f64>) -> Array2<f64> {
        let mut activation = input;
        for (b, w) in self.biases.iter().zip(self.weights.iter()) {
            activation = (w.dot(&activation) + b).mapv(|val| sigmoid(val));
        }
        activation
    }
    pub fn stochastic_gradient_descent(
        &mut self,
        mut training_data: Vec<VectorizedLabelPixelsPair>,
        test_data: Vec<LabelPixelsPair>,
        epochs: u16,
        mini_batch_size: u16,
        eta: f64,
    ) {
        for i in 0..epochs {
            training_data.shuffle(&mut thread_rng());
            let mini_batches = training_data.chunks(mini_batch_size.to_usize().unwrap());
            for mini_batch in mini_batches {
                self.update_mini_batch(mini_batch, eta);
            }
            println!("Epoch {i} complete");
            let correct_classifications_number = self.evaluate(&test_data);
            println!(
                "Accuracy: {}/{} which is {}%",
                correct_classifications_number,
                test_data.len(),
                (correct_classifications_number as f64 / test_data.len() as f64) * 100.0
            );
        }
    }
    fn update_mini_batch(&mut self, mini_batch: &[VectorizedLabelPixelsPair], learning_rate: f64) {
        let mut nabla_b: Vec<Array2<f64>> = Vec::new();
        for bias in self.biases.iter() {
            nabla_b.push(Array2::zeros(bias.raw_dim()));
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
            .map(|(w, nw)| w - (learning_rate / (nw.len() as f64)) * nw)
            .collect();
        self.biases = zip(self.biases.clone(), nabla_b)
            .map(|(b, nb)| b - (learning_rate / (nb.len() as f64)) * nb)
            .collect();
    }
    fn backpropagation(
        &self,
        vector_label_pixels_pair: &VectorizedLabelPixelsPair,
    ) -> (Vec<Array2<f64>>, Vec<Array2<f64>>) {
        let mut nabla_b: Vec<Array2<f64>> = Vec::new();
        for bias in self.biases.iter() {
            nabla_b.push(Array2::zeros(bias.raw_dim()));
        }
        let mut nabla_w: Vec<Array2<f64>> = Vec::new();
        for weight in self.weights.iter() {
            nabla_w.push(Array2::zeros(weight.raw_dim()));
        }

        //feedforward
        let activation = vector_label_pixels_pair.1.clone();
        let mut activations: Vec<Array2<f64>> = vec![activation];
        let mut z_vectors: Vec<Array2<f64>> = Vec::new();
        for (b, w) in zip(self.biases.iter(), self.weights.iter()) {
            let z = w.dot(activations.last().unwrap()) + b;
            z_vectors.push(z.clone());
            let activation = z.mapv(|x| sigmoid(x));
            activations.push(activation);
        }

        //backward pass
        let delta = self.cost_derivative(activations.last().unwrap(), &vector_label_pixels_pair.0)
            * (z_vectors.last().unwrap().map(|x| sigmoid_prime(*x)));

        let nabla_b_len = nabla_b.len();
        nabla_b[nabla_b_len - 1] = delta.clone();

        let nabla_w_len = nabla_w.len();

        let x = activations[activations.len() - 2].t();
        nabla_w[nabla_w_len - 1] = delta.dot(&x);

        for l in 2usize..self.number_of_layers {
            let z = z_vectors[z_vectors.len() - l].clone();
            let sp = z.mapv(|x| sigmoid_prime(x));
            let delta = self.weights[self.weights.len() - l + 1].t().dot(&delta) * sp;
            let nabla_b_len = nabla_b.len();
            nabla_b[nabla_b_len - l] = delta.clone();
            let nabla_w_len = nabla_w.len();
            nabla_w[nabla_w_len - l] = delta.dot(&activations[activations.len() - l - 1].t());
        }
        (nabla_b, nabla_w)
    }
    fn cost_derivative(&self, output_activations: &Array2<f64>, y: &Array2<f64>) -> Array2<f64> {
        output_activations - y
    }
    fn evaluate(&self, test_data: &[LabelPixelsPair]) -> usize {
        let test_results = test_data
            .iter()
            .map(|(label, pixels)| {
                let output = self.feed_forward(pixels.clone());
                let max_index = output
                    .iter()
                    .enumerate()
                    .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap())
                    .unwrap()
                    .0;
                if max_index == label.to_usize().unwrap() {
                    1
                } else {
                    0
                }
            })
            .sum();
        test_results
    }
}
