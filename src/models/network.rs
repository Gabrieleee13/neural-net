use super::DenseLayer;
use super::Matrix;

pub struct Network {
    layers: Vec<DenseLayer>
}

impl Network {
    pub fn new() -> Network {
        Network { 
            layers: vec![] 
        }
    }

    pub fn add_layer(&mut self, layer: DenseLayer) {
        self.layers.push(layer);
    }

    pub fn forward(&self, inputs: &Matrix) -> Matrix {
        let mut current_data = self.layers[0].forward(inputs);
    
        for i in 1..self.layers.len() {
            current_data = self.layers[i].forward(&current_data);
        }
    
        current_data
    }

    pub fn train(&mut self, inputs: &Matrix, target: &Matrix, learning_rate: f64) {
    
        let mut outputs: Vec<Matrix> = vec![];
        
        outputs.push(inputs.clone()); 
    
        for layer in &self.layers {
            let current_input = outputs.last().unwrap(); 
            
            let result = layer.forward(current_input);
            
            outputs.push(result);
        }
    
        let prediction = outputs.last().unwrap();
        
        let mut current_error = target.subtract(prediction);
        
        for i in (0..self.layers.len()).rev() {
            let inputs_layer = outputs[i].clone();
            let outputs_layer = outputs[i + 1].clone();

            current_error = self.layers[i].backward(&inputs_layer, &outputs_layer, &current_error, learning_rate);
        }
    
    }
}