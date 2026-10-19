use crate::{sigmoid, sigmoid_derivative};

use super::Matrix;

pub struct DenseLayer {
    weight: Matrix,
    bias: Matrix
}

impl DenseLayer {
    pub fn new(inputs_size: usize, neurons_count: usize) -> DenseLayer {

        let weight = Matrix::full_of_rands(inputs_size, neurons_count);
        let bias = Matrix::full_of_zero(1, neurons_count);

        DenseLayer {
            weight: weight, 
            bias: bias 
        }

    }

    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        let product = inputs.dot_product(&self.weight);
        let res = product.add(&self.bias);

        res.apply(sigmoid)
    }

    pub fn backward(&mut self, inputs: &Matrix, outputs: &Matrix, output_errors: &Matrix, learning_rate: f64) -> Matrix {
        let mut gradients = outputs.apply(sigmoid_derivative);
        gradients = output_errors.multiply(&gradients);

        let gradients = gradients.apply(|x| x * learning_rate);

        let inputs_t = inputs.transpose();
        let weight_deltas = inputs_t.dot_product(&gradients);

        self.weight = self.weight.add(&weight_deltas);
        self.bias = self.bias.add(&gradients);

        gradients.dot_product(&self.weight.transpose())
    }
}