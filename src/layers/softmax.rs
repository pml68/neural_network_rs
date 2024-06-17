use crate::layers::Layer;
use ndarray::Array2;

pub struct ActivationSoftmax {
    pub output: Option<Array2<f64>>,
}

impl ActivationSoftmax {
    pub fn new() -> Self {
        Self { output: None }
    }

    pub fn forward(&mut self, inputs: Array2<f64>) {
        let mut output_matrix = inputs.clone();

        for i in 0..output_matrix.nrows() {
            let expx = output_matrix.row(i).mapv(f64::exp);
            let probabilities = &expx / expx.sum();

            for j in 0..output_matrix.ncols() {
                output_matrix.row_mut(i)[j] = probabilities[j];
            }
        }

        self.output = Some(output_matrix);
    }
}

impl Layer for ActivationSoftmax {}
