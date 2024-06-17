mod layers;

use layers::dense::DenseLayer;
use layers::relu::ActivationReLU;
use layers::softmax::ActivationSoftmax;
use ndarray::array;

fn main() {
    let inputs = array![[1., 2., 3., 2.5], [2., 5., -1., 2.], [-1.5, 2.7, 3.3, -0.8]];
    let weights = array![
        [0.2, 0.8, -0.5, 1.0],
        [0.5, -0.91, 0.26, -0.5],
        [-0.26, -0.27, 0.17, 0.87],
    ];
    let biases = array![2., 3., 0.5];

    let mut layer1 = DenseLayer {
        weights,
        biases,
        output: None,
    };
    let mut activation1 = ActivationReLU::new();
    let mut layer2 = DenseLayer::new(3, 5);
    let mut activation2 = ActivationSoftmax::new();

    layer1.forward(inputs);
    activation1.forward(layer1.output.unwrap());
    layer2.forward(activation1.output.unwrap());
    activation2.forward(layer2.output.unwrap());

    let output = activation2.output.unwrap();
    println!("{}", output);
}
