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

/// Evaluate model and compute confusion matrix
pub fn evaluate_model(
    model: &CNN,
    val_loader: &mut DataLoader,
    device: Device,
    class_names: &[&str],
) -> anyhow::Result<(Vec<i64>, Vec<i64>)> {
    let mut all_predictions = Vec::new();
    let mut all_labels = Vec::new();
    
    val_loader.reset();
    
    tch::no_grad(|| {
        while let Some((images, labels)) = val_loader.next() {
            let images = images.to_device(device);
            let labels = labels.to_device(device);
            
            // Forward pass
            let outputs = model.forward_t(&images, false);
            let predictions = outputs.argmax(-1, false);
            
            // Convert to CPU and collect
            let pred_cpu = predictions.to_device(Device::Cpu);
            let label_cpu = labels.to_device(Device::Cpu);
            
            // FIXED: Extract values using proper tensor methods
            for i in 0..pred_cpu.size()[0] {
                all_predictions.push(pred_cpu.int64_value(&[i]));
                all_labels.push(label_cpu.int64_value(&[i]));
            }
        }
    });
    
    // Compute accuracy
    let correct = all_predictions.iter().zip(&all_labels)
        .filter(|(p, l)| p == l)
        .count();
    let accuracy = correct as f64 / all_labels.len() as f64;
    
    println!("Validation Accuracy: {:.4} ({}/{})", 
        accuracy, correct, all_labels.len());
    
    // Compute and display confusion matrix
    let confusion_matrix = compute_confusion_matrix(&all_predictions, &all_labels, class_names.len());
    print_confusion_matrix(&confusion_matrix, class_names);
    
    // Compute per-class metrics
    compute_classification_metrics(&confusion_matrix, class_names);
    
    Ok((all_predictions, all_labels))
}

/// Compute confusion matrix
fn compute_confusion_matrix(predictions: &[i64], labels: &[i64], num_classes: usize) -> Vec<Vec<i64>> {
    let mut matrix = vec![vec![0; num_classes]; num_classes];
    
    for (&pred, &label) in predictions.iter().zip(labels.iter()) {
        matrix[label as usize][pred as usize] += 1;
    }
    
    matrix
}

/// Print confusion matrix in a formatted way
fn print_confusion_matrix(matrix: &[Vec<i64>], class_names: &[&str]) {
    println!("\nConfusion Matrix:");
    println!("Predicted ->");
    
    // Header
    print!("Actual\\    ");
    for name in class_names {
        print!("{:>8} ", name);
    }
    println!();
    
    // Matrix rows
    for (i, row) in matrix.iter().enumerate() {
        print!("{:>8} ", class_names[i]);
        for &val in row {
            print!("{:>8} ", val);
        }
        println!();
    }
    println!();
}

/// Compute classification metrics (precision, recall, f1-score)
fn compute_classification_metrics(matrix: &[Vec<i64>], class_names: &[&str]) {
    println!("Classification Metrics:");
    println!("{:>12} {:>10} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1-Score", "Support");
    println!("{}", "-".repeat(60));
    
    let mut total_samples = 0;
    let mut weighted_precision = 0.0;
    let mut weighted_recall = 0.0;
    let mut weighted_f1 = 0.0;
    
    for (i, class_name) in class_names.iter().enumerate() {
        let tp = matrix[i][i];
        let fp: i64 = (0..matrix.len()).filter(|&j| j != i).map(|j| matrix[j][i]).sum();
        let fn_: i64 = (0..matrix[i].len()).filter(|&j| j != i).map(|j| matrix[i][j]).sum();
        let support = matrix[i].iter().sum::<i64>();
        
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_ > 0 { tp as f64 / (tp + fn_) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 { 2.0 * precision * recall / (precision + recall) } else { 0.0 };
        
        println!("{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}", 
            class_name, precision, recall, f1, support);
        
        total_samples += support;
        weighted_precision += precision * support as f64;
        weighted_recall += recall * support as f64;
        weighted_f1 += f1 * support as f64;
    }
    
    // Compute weighted averages
    weighted_precision /= total_samples as f64;
    weighted_recall /= total_samples as f64;
    weighted_f1 /= total_samples as f64;
    
    println!("{}", "-".repeat(60));
    println!("{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}", 
        "Weighted Avg", weighted_precision, weighted_recall, weighted_f1, total_samples);
}

/// Save model weights
pub fn save_model(vs: &nn::VarStore, path: &str) -> anyhow::Result<()> {
    vs.save(path)?;
    println!("Model saved to: {}", path);
    Ok(())
}
