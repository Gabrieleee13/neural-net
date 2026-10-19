mod models;
use std::f64::consts::E;
use crate::models::Network;
use crate::models::DenseLayer;
use crate::models::Matrix;

fn sigmoid(f: f64) -> f64 {
    1.0 / (1.0 + E.powf(-f))
}

fn sigmoid_derivative(f: f64) -> f64 {
    f * (1.0 - f)
}
fn main() {
    let mut network = Network::new();

    network.add_layer(DenseLayer::new(2, 3));
    network.add_layer(DenseLayer::new(3, 1));

    let training_data = vec![
        (
            Matrix { fields: vec![0.0, 0.0], rows: 1, columns: 2 },
            Matrix { fields: vec![0.0], rows: 1, columns: 1 }
        ),
        (
            Matrix { fields: vec![0.0, 1.0], rows: 1, columns: 2 },
            Matrix { fields: vec![1.0], rows: 1, columns: 1 }
        ),
        (
            Matrix { fields: vec![1.0, 0.0], rows: 1, columns: 2 },
            Matrix { fields: vec![1.0], rows: 1, columns: 1 }
        ),
        (
            Matrix { fields: vec![1.0, 1.0], rows: 1, columns: 2 },
            Matrix { fields: vec![0.0], rows: 1, columns: 1 }
        ),
    ];

    let epochs = 10000;
    let learning_rate = 0.1;

    println!("\nTraining started ({} epoch)...\n", epochs);

    for epoch in 0..epochs {
        for (inputs, target) in &training_data {
            network.train(inputs, target, learning_rate);
        }

        if epoch % 2000 == 0 || epoch == epochs - 1 {
            println!("Epoch {} completed", epoch);
        }
    }

    println!("\nThe training hash finished, lets see the result:\n");

    for (inputs, target) in &training_data {
        let prediction = network.forward(inputs);
        
        println!(
            "Input: [{}, {}] -> Prevision: {:.4} (Target: {})",
            inputs.fields[0], 
            inputs.fields[1], 
            prediction.fields[0], 
            target.fields[0]
        );
    }

    println!("\n");
}
