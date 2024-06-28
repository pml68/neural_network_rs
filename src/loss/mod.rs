use ndarray::{Array1, Array2};

pub mod cross_entropy;

pub trait Loss {
    fn new() -> Self;

    fn calculate(&self, output: Array2<f64>, y: Array2<f64>) -> f64 {
        let sample_losses = self.forward(output, y);
        let data_loss = sample_losses.mean().unwrap();
        data_loss
    }

    fn forward(&self, y_pred: Array2<f64>, y_true: Array2<f64>) -> Array1<f64>;
}
