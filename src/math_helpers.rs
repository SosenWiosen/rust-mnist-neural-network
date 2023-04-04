pub(crate) fn sigmoid(z: f64) -> f64 {
    1.0 / (1.0 + (-z).exp())
}
