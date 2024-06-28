use std::f64::consts::E;

use crate::loss::Loss;

use ndarray::{Array1, Array2, Axis};

pub struct LossCrossentropy;

impl Loss for LossCrossentropy {
    fn new() -> Self {
        Self {}
    }

    fn forward(&self, y_pred: Array2<f64>, y_true: Array2<f64>) -> Array1<f64> {
        let y_pred_clipped = y_pred.mapv(|c| c.clamp(1e-7, 1. - 1e-7));

        let correct_confidences = (y_pred_clipped * y_true).sum_axis(Axis(1));

        let neg_log_likelihoods = -correct_confidences.mapv(|c| c.log(E));
        neg_log_likelihoods
    }
}
