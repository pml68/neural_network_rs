use crate::layers::Layer;
use ndarray::{Array2, Axis};

pub struct ActivationSoftmax {
    pub output: Option<Array2<f64>>,
}

impl ActivationSoftmax {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        let expx = inputs.mapv(f64::exp);
        self.output = Some((&expx / expx.sum_axis(Axis(0))).reversed_axes());
    }
}

impl Layer for ActivationSoftmax {}
