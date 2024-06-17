use crate::layers::Layer;
use ndarray::{Array, Array1, Array2};
use ndarray_rand::{rand_distr::Uniform, RandomExt};

pub struct DenseLayer {
    pub weights: Array2<f64>,
    pub biases: Array1<f64>,
    pub output: Option<Array2<f64>>,
}

impl DenseLayer {
    pub fn new(n_inputs: usize, n_neurons: usize) -> Self {
        Self {
            weights: Array::random((n_neurons, n_inputs), Uniform::new(0., 1.)),
            biases: Array::from_vec(vec![0.; n_neurons]),
            output: None,
        }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        self.output = Some(inputs.dot(&self.weights.t()) + &self.biases);
    }
}

impl Layer for DenseLayer {}
