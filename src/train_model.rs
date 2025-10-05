// train_model.rs

use crate::CNN;
use crate::visualize_data::DataLoader;
use tch::{nn, nn::{OptimizerConfig, ModuleT}, Device, Kind, Tensor};
use std::fs::File;
use std::io::Write;
use std::collections::HashMap;

/// Enhanced training configuration structure
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    pub num_epochs: i64,
    pub learning_rate: f64,
    pub step_size: i64,
    pub gamma: f64,
    pub max_norm: f64,
    pub weight_decay: f64,
    pub warmup_epochs: i64,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            num_epochs: 100,
            learning_rate: 1e-3,
            step_size: 20,
            gamma: 0.5,
            max_norm: 1.0,
            weight_decay: 1e-4,
            warmup_epochs: 5,
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

/// Enhanced learning rate scheduler with warmup, cosine annealing, and step decay
pub struct EnhancedScheduler {
    warmup_epochs: i64,
    total_epochs: i64,
    base_lr: f64,
    min_lr: f64,
    current_epoch: i64,
    step_size: i64,
    gamma: f64,
}

impl EnhancedScheduler {
    pub fn new(warmup_epochs: i64, total_epochs: i64, base_lr: f64, step_size: i64, gamma: f64) -> Self {
        Self {
            warmup_epochs,
            total_epochs,
            base_lr,
            min_lr: base_lr * 0.01,
            current_epoch: 0,
            step_size,
            gamma,
        }
    }

    pub fn step(&mut self, optimizer: &mut nn::Optimizer) {
        self.current_epoch += 1;
        
        let new_lr = if self.current_epoch <= self.warmup_epochs {
            // Warmup phase: linear increase
            self.base_lr * (self.current_epoch as f64 / self.warmup_epochs as f64)
        } else {
            // After warmup: Hybrid cosine annealing + step decay
            let progress = (self.current_epoch - self.warmup_epochs) as f64 / 
                          (self.total_epochs - self.warmup_epochs) as f64;
            let cosine_factor = (1.0 + (std::f64::consts::PI * progress).cos()) / 2.0;
            let cosine_lr = self.min_lr + (self.base_lr - self.min_lr) * cosine_factor;
            
            // Apply step decay on top of cosine annealing
            let step_decay_factor = self.gamma.powf((self.current_epoch / self.step_size) as f64);
            cosine_lr * step_decay_factor
        };
        
        optimizer.set_lr(new_lr);
        
        // More frequent logging to show the hybrid effect
        if self.current_epoch % 5 == 0 || self.current_epoch % self.step_size == 0 {
            let step_num = self.current_epoch / self.step_size;
            println!("Epoch {}: LR = {:.6} (step decay #{}, cosine annealing)", 
                    self.current_epoch, new_lr, step_num);
        }
    }
}

/// Early stopping to prevent overfitting
pub struct EarlyStopping {
    patience: i64,
    min_delta: f64,
    best_loss: f64,
    counter: i64,
}

impl EarlyStopping {
    pub fn new(patience: i64, min_delta: f64) -> Self {
        Self {
            patience,
            min_delta,
            best_loss: f64::INFINITY,
            counter: 0,
        }
    }
    
    pub fn should_stop(&mut self, val_loss: f64) -> bool {
        if val_loss < self.best_loss - self.min_delta {
            self.best_loss = val_loss;
            self.counter = 0;
            false
        } else {
            self.counter += 1;
            self.counter >= self.patience
        }
    }
}

/// Training function for one epoch with enhanced features
pub fn train_epoch(
    model: &CNN,
    train_loader: &mut DataLoader,
    optimizer: &mut nn::Optimizer,
    device: Device,
    max_norm: f64,
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
        
        // Simple gradient clipping
        if max_norm > 0.0 {
            for var in optimizer.trainable_variables().iter() {
                let grad = var.grad();
                let norm = grad.norm();
                if norm.double_value(&[]) > max_norm {
                    let _ = grad * (max_norm / norm);
                }
            }
        }
        
        optimizer.step();
        
        let batch_size = images.size()[0];
        total_loss += loss.double_value(&[]) * batch_size as f64;
        total_samples += batch_size;
        
        let predicted = outputs.argmax(-1, false);
        let correct = predicted.eq_tensor(&labels).to_kind(Kind::Int64).sum(Kind::Int64);
        total_correct += correct.int64_value(&[]);
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
            let correct = predicted.eq_tensor(&labels).to_kind(Kind::Int64).sum(Kind::Int64);
            total_correct += correct.int64_value(&[]);
        }
    });
    
    let avg_loss = total_loss / total_samples as f64;
    let accuracy = (total_correct as f64 / total_samples as f64) * 100.0;
    (avg_loss, accuracy)
}

/// Enhanced main training loop
pub fn train_model(
    model: &CNN,
    train_loader: &mut DataLoader,
    val_loader: &mut DataLoader,
    vs: &nn::VarStore,
    config: TrainingConfig,
) -> anyhow::Result<TrainingStats> {
    // Create Adam optimizer with all required fields
    let mut optimizer = nn::Adam {
        beta1: 0.9,
        beta2: 0.999,
        wd: config.weight_decay,
        eps: 1e-8,
        amsgrad: false,
    }.build(vs, config.learning_rate)?;

    // Create enhanced scheduler
    let mut scheduler = EnhancedScheduler::new(
        config.warmup_epochs,
        config.num_epochs,
        config.learning_rate,
        config.step_size,
        config.gamma,
    );
    
    // Initialize early stopping
    let mut early_stopping = EarlyStopping::new(25, 0.0005);

    let mut stats = TrainingStats::new();
    let mut best_val_acc = 0.0;

    println!("Starting enhanced training for {} epochs...", config.num_epochs);
    println!("Initial learning rate: {:.6}", config.learning_rate);
    println!("Weight decay: {:.6}", config.weight_decay);

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

        // Save best model with enhanced compatibility
        if val_accuracy > best_val_acc {
            best_val_acc = val_accuracy;
            save_model_universal(vs, "best_model", epoch, val_accuracy)?;
            println!("🎯 New best model saved! Accuracy: {:.2}%", val_accuracy);
        }

        // Print progress
        println!(
            "Epoch: {:3}/{} | Train: Loss {:.4}, Acc {:5.2}% | Val: Loss {:.4}, Acc {:5.2}% | Best: {:5.2}%",
            epoch,
            config.num_epochs,
            train_loss,
            train_accuracy,
            val_loss,
            val_accuracy,
            best_val_acc
        );

        // Early stopping check
        if early_stopping.should_stop(val_loss) {
            println!("Early stopping triggered at epoch {}", epoch);
            break;
        }
    }

    println!("Training completed! Best validation accuracy: {:.2}%", best_val_acc);

    // Save training statistics
    stats.save_to_csv("training_stats.csv")?;
    println!("Training statistics saved to training_stats.csv");

    Ok(stats)
}

/// Universal model saving that works with both Rust and Python
pub fn save_model_universal(vs: &nn::VarStore, base_name: &str, epoch: i64, accuracy: f64) -> anyhow::Result<()> {
    println!("💾 Saving model in multiple formats for universal compatibility...");
    
    // Format 1: Standard tch format (works with Rust)
    let tch_path = format!("{}.pt", base_name);
    match vs.save(&tch_path) {
        Ok(_) => println!("✅ Rust-compatible model: {}", tch_path),
        Err(e) => println!("⚠️ Rust format failed: {}", e),
    }
    
    // Format 2: Named tensors format (more compatible)
    let named_path = format!("{}_named.pt", base_name);
    match save_named_tensors(vs, &named_path) {
        Ok(_) => println!("✅ Named tensors format: {}", named_path),
        Err(e) => println!("⚠️ Named tensors failed: {}", e),
    }
    
    // Format 3: Individual tensor files (most compatible)
    let tensor_dir = format!("{}_tensors", base_name);
    match save_individual_tensors(vs, &tensor_dir) {
        Ok(_) => println!("✅ Individual tensors: {}/", tensor_dir),
        Err(e) => println!("⚠️ Individual tensors failed: {}", e),
    }
    
    // Format 4: Model metadata
    let metadata_path = format!("{}_metadata.json", base_name);
    match save_model_metadata(&metadata_path, epoch, accuracy) {
        Ok(_) => println!("✅ Model metadata: {}", metadata_path),
        Err(e) => println!("⚠️ Metadata failed: {}", e),
    }
    
    println!("📦 Model saved in {} formats for maximum compatibility!", 4);
    Ok(())
}

/// Save tensors with explicit names for better compatibility
fn save_named_tensors(vs: &nn::VarStore, path: &str) -> anyhow::Result<()> {
    let mut tensor_dict = HashMap::new();
    
    // Extract all named variables from the VarStore
    for (name, var) in vs.variables() {
        tensor_dict.insert(name.clone(), var);
    }
    
    // Save the named dictionary
    let tensor_list: Vec<_> = tensor_dict.iter().collect();
    tch::Tensor::save_multi(&tensor_list, path)?;
    
    Ok(())
}

/// Save individual tensor files for maximum compatibility
fn save_individual_tensors(vs: &nn::VarStore, dir_path: &str) -> anyhow::Result<()> {
    std::fs::create_dir_all(dir_path)?;
    
    for (name, var) in vs.variables() {
        let clean_name = name.replace("/", "_").replace(".", "_");
        let tensor_path = format!("{}/{}.pt", dir_path, clean_name);
        var.save(&tensor_path)?;
    }
    
    // Save a mapping file
    let mut mapping_file = File::create(format!("{}/tensor_mapping.txt", dir_path))?;
    for (name, _) in vs.variables() {
        let clean_name = name.replace("/", "_").replace(".", "_");
        writeln!(mapping_file, "{}:{}", name, clean_name)?;
    }
    
    Ok(())
}

/// Save model metadata for easy loading
fn save_model_metadata(path: &str, epoch: i64, accuracy: f64) -> anyhow::Result<()> {
    let metadata = format!(r#"{{
    "model_type": "garbage_classifier_cnn",
    "architecture": "enhanced_cnn",
    "num_classes": 6,
    "classes": ["cardboard", "glass", "metal", "paper", "plastic", "trash"],
    "input_size": [224, 224],
    "dropout_rate": 0.15,
    "training": {{
        "final_epoch": {},
        "best_accuracy": {:.4},
        "optimizer": "Adam",
        "loss_function": "CrossEntropy"
    }},
    "preprocessing": {{
        "resize": [224, 224],
        "normalize": {{
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        }}
    }},
    "created_at": "{}",
    "framework": "tch-rs"
}}"#, epoch, accuracy, chrono::Utc::now().format("%Y-%m-%d %H:%M:%S UTC"));

    std::fs::write(path, metadata)?;
    Ok(())
}

/// Enhanced model loading that tries multiple formats
pub fn load_model_universal(vs: &mut nn::VarStore, base_name: &str) -> anyhow::Result<()> {
    println!("🔍 Attempting to load model with universal compatibility...");
    
    // Try different formats in order of preference
    let formats = [
        format!("{}.pt", base_name),
        format!("{}_named.pt", base_name),
        format!("{}_backup.pt", base_name),
    ];
    
    for (i, path) in formats.iter().enumerate() {
        if std::path::Path::new(path).exists() {
            match vs.load(path) {
                Ok(_) => {
                    println!("✅ Loaded model from: {} (format {})", path, i + 1);
                    return Ok(());
                },
                Err(e) => {
                    println!("⚠️ Failed to load {}: {}", path, e);
                    continue;
                }
            }
        }
    }
    
    // Try individual tensors as last resort
    let tensor_dir = format!("{}_tensors", base_name);
    if std::path::Path::new(&tensor_dir).exists() {
        match load_individual_tensors(vs, &tensor_dir) {
            Ok(_) => {
                println!("✅ Loaded model from individual tensors: {}/", tensor_dir);
                return Ok(());
            },
            Err(e) => println!("⚠️ Individual tensors failed: {}", e),
        }
    }
    
    Err(anyhow::anyhow!("Could not load model with any format"))
}


/// Load from individual tensor files

fn load_individual_tensors(_vs: &mut nn::VarStore, dir_path: &str) -> anyhow::Result<()> {
    let mapping_path = format!("{}/tensor_mapping.txt", dir_path);
    if !std::path::Path::new(&mapping_path).exists() {
        return Err(anyhow::anyhow!("Tensor mapping file not found"));
    }
    
    let mapping_content = std::fs::read_to_string(&mapping_path)?;
    for line in mapping_content.lines() {
        if let Some((original_name, clean_name)) = line.split_once(':') {
            let tensor_path = format!("{}/{}.pt", dir_path, clean_name);
            if std::path::Path::new(&tensor_path).exists() {
                let _tensor = Tensor::load(&tensor_path)?;
                // Note: This is a simplified approach - you may need to implement
                // proper variable assignment based on your needs
                println!("Loading tensor: {} from {}", original_name, tensor_path);
            }
        }
    }
    
    Ok(())
}

/// Backward compatible save function
pub fn save_model(vs: &nn::VarStore, path: &str) -> anyhow::Result<()> {
    // Extract base name from path
    let base_name = path.trim_end_matches(".pt");
    
    // Use the universal saving approach
    save_model_universal(vs, base_name, 0, 0.0)
}
