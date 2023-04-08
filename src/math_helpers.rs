pub fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
pub fn sigmoid_prime(z: f64) -> f64 {
    sigmoid(z) * (1.0 - sigmoid(z))
}
