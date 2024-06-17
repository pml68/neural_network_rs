use crate::layers::Layer;
use ndarray::Array2;

pub struct ActivationReLU {
    pub output: Option<Array2<f64>>,
}

fn relu(n: f64) -> f64 {
    if n < 0. {
        0.
    } else {
        n
    }
}

impl ActivationReLU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.output = Some(inputs.mapv(|c| relu(c)));
    }
}

impl Layer for ActivationReLU {}
