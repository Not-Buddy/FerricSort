// train_model.rs

use crate::CNN;
use crate::visualize_data::DataLoader;
use tch::{nn, nn::{OptimizerConfig, ModuleT}, Device, Kind};
use std::fs::File;
use std::io::Write;

/// Training configuration structure
pub struct TrainingConfig {
    pub num_epochs: i64,
    pub learning_rate: f64,
    pub step_size: i64,
    pub gamma: f64,
    pub max_norm: f64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            learning_rate: 1e-4,
            step_size: 5,
            gamma: 0.85,
            max_norm: 1.0,
        }
    }
}

/// Training statistics to track performance
pub struct TrainingStats {
    pub training_losses: Vec<f64>,
    pub validation_losses: Vec<f64>,
    pub training_accuracies: Vec<f64>,
    pub validation_accuracies: Vec<f64>,
}

impl TrainingStats {
    pub fn new() -> Self {
        Self {
            training_losses: Vec::new(),
            validation_losses: Vec::new(),
            training_accuracies: Vec::new(),
            validation_accuracies: Vec::new(),
        }
    }

    /// Save training metrics to CSV file
    pub fn save_to_csv(&self, filename: &str) -> std::io::Result<()> {
        let mut file = File::create(filename)?;
        writeln!(file, "epoch,train_loss,train_accuracy,val_loss,val_accuracy")?;
        
        for i in 0..self.training_losses.len() {
            writeln!(file, "{},{:.6},{:.4},{:.6},{:.4}", 
                i + 1,
                self.training_losses[i],
                self.training_accuracies[i],
                self.validation_losses[i],
                self.validation_accuracies[i]
            )?;
        }
        Ok(())
    }
}

/// Training function for one epoch
/// Training function for one epoch
pub fn train_epoch(
    model: &CNN,
    train_loader: &mut DataLoader,
    optimizer: &mut nn::Optimizer,
    device: Device,
    _max_norm: f64,
) -> (f64, f64) {
    let mut total_loss = 0.0;
    let mut total_samples = 0;
    let mut total_correct = 0;
    
    train_loader.reset();
    
    while let Some((images, labels)) = train_loader.next() {
        let images = images.to_device(device);
        let labels = labels.to_device(device);
        
        optimizer.zero_grad();
        let outputs = model.forward_t(&images, true);
        let loss = outputs.cross_entropy_for_logits(&labels);
        loss.backward();
        optimizer.step();
        
        let batch_size = images.size()[0];
        total_loss += loss.double_value(&[]) * batch_size as f64;
        total_samples += batch_size;
        
        let predicted = outputs.argmax(-1, false);
        let correct = predicted.eq_tensor(&labels).to_kind(Kind::Int64).sum(Kind::Int64);  // FIXED
        total_correct += correct.int64_value(&[]);  // FIXED
    }
    
    let avg_loss = total_loss / total_samples as f64;
    let accuracy = (total_correct as f64 / total_samples as f64) * 100.0;
    
    (avg_loss, accuracy)
}

/// Validation function
pub fn validate_epoch(
    model: &CNN,
    val_loader: &mut DataLoader,
    device: Device,
) -> (f64, f64) {
    let mut total_loss = 0.0;
    let mut total_samples = 0;
    let mut total_correct = 0;
    
    val_loader.reset();
    
    tch::no_grad(|| {
        while let Some((images, labels)) = val_loader.next() {
            let images = images.to_device(device);
            let labels = labels.to_device(device);
            
            let outputs = model.forward_t(&images, false);
            let loss = outputs.cross_entropy_for_logits(&labels);
            
            let batch_size = images.size()[0];
            total_loss += loss.double_value(&[]) * batch_size as f64;
            total_samples += batch_size;
            
            let predicted = outputs.argmax(-1, false);
            let correct = predicted.eq_tensor(&labels).to_kind(Kind::Int64).sum(Kind::Int64);  // FIXED
            total_correct += correct.int64_value(&[]);  // FIXED
        }
    });
    
    let avg_loss = total_loss / total_samples as f64;
    let accuracy = (total_correct as f64 / total_samples as f64) * 100.0;
    
    (avg_loss, accuracy)
}


/// Simple learning rate scheduler (StepLR equivalent)
pub struct StepLR {
    step_size: i64,
    gamma: f64,
    current_epoch: i64,
    base_lr: f64,
}

impl StepLR {
    pub fn new(step_size: i64, gamma: f64, base_lr: f64) -> Self {
        Self {
            step_size,
            gamma,
            current_epoch: 0,
            base_lr,
        }
    }

    pub fn step(&mut self, optimizer: &mut nn::Optimizer) {
        self.current_epoch += 1;
        
        if self.current_epoch % self.step_size == 0 {
            let new_lr = self.base_lr * self.gamma.powi((self.current_epoch / self.step_size) as i32);
            optimizer.set_lr(new_lr);
            println!("Learning rate updated to: {:.6}", new_lr);
        }
    }
}

/// Main training loop
pub fn train_model(
    model: &CNN,
    train_loader: &mut DataLoader,
    val_loader: &mut DataLoader,
    vs: &nn::VarStore,
    config: TrainingConfig,
) -> anyhow::Result<TrainingStats> {
    // Create optimizer (Adam equivalent)
    let mut optimizer = nn::Adam::default().build(vs, config.learning_rate)?;
    
    // Create learning rate scheduler
    let mut scheduler = StepLR::new(config.step_size, config.gamma, config.learning_rate);
    
    // Initialize statistics tracking
    let mut stats = TrainingStats::new();
    
    println!("Starting training for {} epochs...", config.num_epochs);
    println!("Initial learning rate: {:.6}", config.learning_rate);
    
    for epoch in 1..=config.num_epochs {
        // Training phase
        let (train_loss, train_accuracy) = train_epoch(
            model,
            train_loader,
            &mut optimizer,
            vs.device(),
            config.max_norm,
        );
        
        // Learning rate scheduling
        scheduler.step(&mut optimizer);
        
        // Validation phase
        let (val_loss, val_accuracy) = validate_epoch(
            model,
            val_loader,
            vs.device(),
        );
        
        // Store statistics
        stats.training_losses.push(train_loss);
        stats.training_accuracies.push(train_accuracy);
        stats.validation_losses.push(val_loss);
        stats.validation_accuracies.push(val_accuracy);
        
        // Print progress
        println!(
            "Epoch: {}/{} | Train Loss: {:.4}, Train Accuracy: {:.2}% | Val Loss: {:.4}, Val Accuracy: {:.2}%",
            epoch,
            config.num_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy
        );
    }
    
    println!("Training completed!");
    
    // Save training statistics
    stats.save_to_csv("training_stats.csv")?;
    println!("Training statistics saved to training_stats.csv");
    
    Ok(stats)
}

/// Save model weights
pub fn save_model(vs: &nn::VarStore, path: &str) -> anyhow::Result<()> {
    vs.save(path)?;
    println!("Model saved to: {}", path);
    Ok(())
}
