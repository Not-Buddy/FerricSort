// main.rs

mod visualize_data;
mod relu;
mod train_model;
mod menu;
mod training_validation;

use relu::CNN;
use menu::{Sample, load_dataset};
use std::{env, io::{self, Write}};
use tch::{Device, nn, vision, Tensor, Kind};
use tch::nn::ModuleT;
use train_model::{TrainingConfig, train_model, save_model, load_model_universal};
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
        println!(" 5 - 🔍 Manual Model Testing");
        println!("Example: cargo run -- 5");
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
            Some((usize::MAX, 100, 1e-3, 10, 0.8, 32, Device::cuda_if_available()))
        },
        "4" => {
            println!("🎯 Enhanced Custom Configuration");
            Some((3000, 50, 1e-3, 8, 0.85, 24, Device::cuda_if_available()))
        },
        _ => {
            println!("❌ Invalid choice! Use 1, 2, 3, 4, or 5");
            None
        }
    }
}

fn test_single_image(
    model: &CNN,
    image_path: &str,
    class_names: &[&str],
    device: Device,
) -> anyhow::Result<()> {
    if !std::path::Path::new(image_path).exists() {
        println!("❌ File not found: {}", image_path);
        return Ok(());
    }
    
    println!("📷 Processing: {}", image_path);
    
    // Load and preprocess the image
    let image = vision::image::load(image_path)?;
    
    // Resize to 224x224 (same as training)
    let resized = vision::image::resize(&image, 224, 224)?;
    
    // Normalize (ImageNet normalization - same as training)
    let mean = Tensor::from_slice(&[0.485, 0.456, 0.406]).to_device(device).view([3, 1, 1]);
    let std = Tensor::from_slice(&[0.229, 0.224, 0.225]).to_device(device).view([3, 1, 1]);
    let normalized = ((resized.to_device(device) / 255.0) - &mean) / &std;
    let batch = normalized.unsqueeze(0);
    
    // Get prediction
    tch::no_grad(|| {
        let output = model.forward_t(&batch, false);
        let probabilities = output.softmax(-1, Kind::Float);
        
        println!("🎯 Predictions:");
        println!("{:-<50}", "");
        
        // Get sorted predictions
        let (sorted_probs, sorted_indices) = probabilities.sort(-1, true);
        
        // Emojis for ranking
        let emojis = ["🏆", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣"];
        
        for i in 0..class_names.len() {
            let class_idx = sorted_indices.int64_value(&[0, i as i64]) as usize; // FIXED: Type casting
            let confidence = sorted_probs.double_value(&[0, i as i64]) * 100.0;   // FIXED: Type casting
            let emoji = if i < emojis.len() { emojis[i] } else { "  " };
            
            // Create visual bar
            let bar_length = (confidence / 5.0) as usize;
            let bar = "█".repeat(bar_length);
            
            println!("{} {:<10} {:6.1}% {}", emoji, class_names[class_idx], confidence, bar);
        }
        
        println!("{:-<50}", "");
        
        // Show the top prediction prominently
        let top_class_idx = sorted_indices.int64_value(&[0, 0]) as usize;
        let top_confidence = sorted_probs.double_value(&[0, 0]) * 100.0;
        
        println!("🎯 PREDICTION: {} ({:.1}% confidence)", 
                 class_names[top_class_idx].to_uppercase(), top_confidence);
    });
    
    Ok(())
}

fn test_model_interactive() -> anyhow::Result<()> {
    println!("🔍 Loading trained model for testing...");
    
    // Try to find a trained model (now with multiple format support)
    let model_candidates = [
        "best_model",
        "garbage_classifier_48_epochs_2700_samples",
        "garbage_classifier_50_epochs_2700_samples",
    ];
    
    let mut found_model = None;
    for base_name in &model_candidates {
        // Check if any format exists
        let formats = [
            format!("{}.pt", base_name),
            format!("{}_named.pt", base_name),
            format!("{}_tensors", base_name),
        ];
        
        for format_path in &formats {
            if std::path::Path::new(format_path).exists() {
                found_model = Some(*base_name);
                break;
            }
        }
        
        if found_model.is_some() {
            break;
        }
    }
    
    let model_base_name = match found_model {
        Some(name) => name,
        None => {
            println!("❌ No trained model found!");
            println!("Please train a model first using options 1-4");
            println!("Looking for files like:");
            println!("  - best_model.pt");
            println!("  - best_model_named.pt");
            println!("  - best_model_tensors/");
            return Ok(());
        }
    };
    
    // Setup device and load model
    let device = Device::cuda_if_available();
    let mut vs = nn::VarStore::new(device); // Make it mutable for loading
    let model = CNN::new(&vs.root(), 6, 0.15);
    
    // Use enhanced universal loading
    match load_model_universal(&mut vs, model_base_name) {
        Ok(_) => {
            println!("✅ Successfully loaded model: {}", model_base_name);
        },
        Err(e) => {
            println!("❌ Error loading model: {}", e);
            println!("💡 Trying fallback to untrained model for testing...");
            println!("⚠️ Predictions will be random!");
            // Continue with untrained model
        }
    }
    
    let class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];
    
    match device {
        Device::Cpu => println!("🖥️ Using CPU for inference"),
        Device::Cuda(_) => println!("🚀 Using GPU for inference"),
        _ => println!("⚠️ Using alternative device for inference"),
    }
    
    println!("\n🎯 Manual Image Testing");
    println!("Enter image paths to classify (or 'quit' to exit):");
    
    loop {
        print!("Image path: ");
        io::stdout().flush()?; 
        
        let mut input = String::new();
        io::stdin().read_line(&mut input)?;
        let input = input.trim();
        
        if input.is_empty() {
            continue;
        }
        
        if input.eq_ignore_ascii_case("quit") || input.eq_ignore_ascii_case("exit") || input.eq_ignore_ascii_case("q") {
            break;
        }
        
        match test_single_image(&model, input, &class_names, device) { 
            Ok(_) => {},
            Err(e) => println!("❌ Error processing image: {}", e),
        }
        
        println!(); // Empty line for readability
    }
    
    println!("👋 Testing session ended!");
    Ok(())
}

fn main() -> anyhow::Result<()> {
    let choice = match parse_config_choice() {
        Some(c) => c,
        None => return Ok(()),
    };

    if choice == "5" {
        return test_model_interactive();
    }

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

    // Training phase
    println!("\n🏋️ Starting Training...");
    let config = TrainingConfig {
        num_epochs,
        learning_rate,
        step_size,
        gamma,
        max_norm: 1.0,
        weight_decay: 1e-4,
        warmup_epochs: 5,
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
