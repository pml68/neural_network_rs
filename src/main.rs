mod layers;
mod loss;

use layers::dense::DenseLayer;
use layers::relu::ActivationReLU;
use layers::softmax::ActivationSoftmax;
use loss::{cross_entropy::LossCrossentropy, Loss};
use ndarray::array;

fn main() {
    let inputs = array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, -0.8, 3.3]];
    let weights = array![
        [0.2, 0.8, -0.5, 1., 2., 5., -1., 2.],
        [0.5, -0.91, 0.26, -0.5, -1.5, 2.7, 3.3, -0.8],
        [3.16, 5.86, -7.32, 3.6, 6.12, -7.63, 3.04, -3.49],
        [0.2, -0.91, -7.32, 1., -1.5, -7.63, -1., 3.3]
    ];
    let biases = array![2., 0.5, -1.8, -3.49];

    let y = array![[0., 0., 1., 0.], [0., 1., 0., 0.], [0., 0., 0., 1.]];

    let mut layer1 = DenseLayer::new(4, 8);
    let mut activation1 = ActivationReLU::new();
    let mut layer2 = DenseLayer::new(8, 8);
    let mut activation2 = ActivationReLU::new();
    let mut layer3 = DenseLayer::new(8, 4); //{
                                            //    weights,
                                            //    biases,
                                            //    output: None,
                                            //};
    let mut activation3 = ActivationSoftmax::new();

    layer1.forward(inputs);
    activation1.forward(layer1.output.unwrap());
    layer2.forward(activation1.output.unwrap());
    activation2.forward(layer2.output.unwrap());
    layer3.forward(activation2.output.unwrap());
    activation3.forward(layer3.output.unwrap());

    let loss_function = LossCrossentropy::new();
    let loss = loss_function.calculate(activation3.output.clone().unwrap(), y);

    let output = activation3.output.unwrap();
    println!("{}", output);
    println!("Loss: {}", loss);
}
