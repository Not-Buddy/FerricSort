// main.rs
mod visualize_data;
mod relu;
mod train_model;
mod menu;
mod training_validation;

use relu::CNN;
use menu::{Sample, load_dataset};
use std::{env};
use tch::{Device, nn};
use train_model::{TrainingConfig, train_model, save_model};
use visualize_data::{DataLoader, visualize_batch};
use training_validation::run_complete_evaluation;

fn parse_config_choice() -> Option<String> {
    let args: Vec<String> = env::args().collect();
    if args.len() < 2 {
        println!("Usage: cargo run -- <choice>");
        println!("Available choices:");
        println!(" 1 - Quick Test (500 samples, 2 epochs, CPU)");
        println!(" 2 - Medium Test (2000 samples, 5 epochs, CPU/GPU)");
        println!(" 3 - Full Training (All samples, 100 epochs, GPU)");
        println!(" 4 - Custom Configuration");
        println!("Example: cargo run -- 2");
        return None;
    }

    Some(args[1].clone())
}

fn get_training_config_cli(choice: &str) -> Option<(usize, i64, f64, i64, f64, i64, Device)> {
    match choice {
        "1" => {
            println!("🚀 Quick Test Configuration Selected");
            Some((500, 2, 1e-3, 1, 0.9, 8, Device::Cpu))
        },
        "2" => {
            println!("🔬 Medium Test Configuration Selected");
            Some((2000, 5, 1e-3, 2, 0.85, 16, Device::cuda_if_available()))
        },
        "3" => {
            println!("📊 Full Training Configuration Selected - ENHANCED");
            Some((usize::MAX, 100, 1e-3, 10, 0.8, 32, Device::cuda_if_available()))  // Updated LR
        },
        "4" => {
            println!("🎯 Enhanced Custom Configuration");
            Some((3000, 50, 1e-3, 8, 0.85, 24, Device::cuda_if_available()))
        },
        _ => {
            println!("❌ Invalid choice! Use 1, 2, 3, or 4");
            None
        }
    }
}

fn main() -> anyhow::Result<()> {
    let choice = match parse_config_choice() {
        Some(c) => c,
        None => return Ok(()),
    };

    let config = match get_training_config_cli(&choice) {
        Some(config) => config,
        None => return Ok(()),
    };

    let (max_samples, num_epochs, learning_rate, step_size, gamma, batch_size, device) = config;

    println!("\n🔄 Loading dataset...");
    let root_dir = "data/Garbage_Dataset_Classification/images";
    let mut data = load_dataset(root_dir);

    // Limit dataset size if requested
    if max_samples < data.len() {
        data = data.into_iter().take(max_samples).collect();
        println!("📉 Limited dataset to {} samples", max_samples);
    }

    // Train/val split (80/20)
    let split = (data.len() as f64 * 0.8) as usize;
    let (train_data, val_data) = data.split_at(split);

    let mut train_loader = DataLoader::new(train_data, batch_size as usize, true);
    let mut val_loader = DataLoader::new(val_data, batch_size as usize, false);

    // Create model
    let vs = nn::VarStore::new(device);
    let num_classes = 6;
    let dropout_rate = 0.15;
    let model = CNN::new(&vs.root(), num_classes, dropout_rate);

    println!("✅ Created CNN model with {} classes", num_classes);

    // Device info
    match device {
        Device::Cpu => println!("🖥️ Using CPU for training"),
        Device::Cuda(_) => println!("🚀 Using GPU for training"),
        _ => println!("⚠️ Using alternative device for training"),
    }

    // Visualize first batch
    visualize_batch(&mut val_loader, "val_batch_grid.png")?;

    println!("\n📊 Dataset Information:");
    println!(" Total samples: {}", data.len());
    println!(" Training samples: {}", train_data.len());
    println!(" Validation samples: {}", val_data.len());
    println!(" Batch size: {}", batch_size);
    println!(" First sample tensor shape: {:?}", data[0].img.size());
    println!(" Sample tensor range: {:.3} to {:.3}",
        data[0].img.min().double_value(&[]),
        data[0].img.max().double_value(&[]));

    // Training phase - FIXED: Added missing fields
    println!("\n🏋️ Starting Training...");
    let config = TrainingConfig {
        num_epochs,
        learning_rate,
        step_size,
        gamma,
        max_norm: 1.0,
        weight_decay: 1e-4,        // FIXED: Added missing field
        warmup_epochs: 5,          // FIXED: Added missing field
    };

    println!("⚙️ Training Configuration:");
    println!(" Epochs: {}", num_epochs);
    println!(" Learning Rate: {:.6}", learning_rate);
    println!(" Step Size (decay interval): {}", step_size);
    println!(" Gamma (decay factor): {}", gamma);
    println!(" Weight Decay: {:.6}", config.weight_decay);
    println!(" Warmup Epochs: {}", config.warmup_epochs);
    println!(" 🔄 Using Hybrid: Cosine Annealing + Step Decay");


    let start_time = std::time::Instant::now();

    let _stats = train_model(&model, &mut train_loader, &mut val_loader, &vs, config)?;

    let training_duration = start_time.elapsed();
    println!("⏱️ Training completed in: {:?}", training_duration);

    // Comprehensive evaluation
    let class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

    let train_losses: Vec<f64> = _stats.training_losses.clone();
    let val_losses: Vec<f64> = _stats.validation_losses.clone();

    run_complete_evaluation(
        &model,
        &mut val_loader,
        device,
        &class_names,
        &train_losses,
        &val_losses,
    )?;

    // Save model
    let model_name = format!("garbage_classifier_{}_epochs_{}_samples.pt", num_epochs, data.len());
    save_model(&vs, &model_name)?;

    println!("\n🎉 Training and evaluation completed successfully!");
    println!("💾 Model saved as {}", model_name);
    println!("⏱️ Total time: {:?}", training_duration);

    Ok(())
}
