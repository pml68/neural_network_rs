use crate::layers::Layer;
use ndarray::Array2;

pub struct ActivationReLU {
    pub output: Option<Array2<f64>>,
}

impl ActivationReLU {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.output = Some(inputs.mapv(|c| if c < 0. { 0. } else { c }));
    }
}

impl Layer for ActivationReLU {}
