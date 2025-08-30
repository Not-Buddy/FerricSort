// main.rs
mod visualize_data;
mod relu;
mod train_model;
mod menu;
mod training_validation; // Add this module

use relu::CNN;
use menu::{Sample, load_dataset}; 
use std::io::Write;
use tch::{Device, nn};
use train_model::{TrainingConfig, train_model, save_model};
use visualize_data::{DataLoader, visualize_batch};
use training_validation::run_complete_evaluation;
fn main() -> anyhow::Result<()> {
    loop {
        menu::display_menu();
        let choice = menu::get_user_input();

        if choice == "5" {
            println!("👋 Exiting...");
            break;
        }

        let config = match menu::get_training_config(&choice) {
            Some(config) => config,
            None => {
                println!("❌ Invalid choice! Please try again.");
                continue;
            }
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

        // Train/val split example (80/20 split)
        let split = (data.len() as f64 * 0.8) as usize;
        let (train_data, val_data) = data.split_at(split);

        let mut train_loader = DataLoader::new(train_data, batch_size as usize, true);
        let mut val_loader = DataLoader::new(val_data, batch_size as usize, false);

        // ── CREATE MODEL HERE ────────────────────────────────────────────────
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

        // Visualize the first batch from val_loader
        visualize_batch(&mut val_loader, "val_batch_grid.png")?;

        println!("\n📊 Dataset Information:");
        println!("  Total samples: {}", data.len());
        println!("  Training samples: {}", train_data.len());
        println!("  Validation samples: {}", val_data.len());
        println!("  Batch size: {}", batch_size);
        println!("  First sample tensor shape: {:?}", data[0].img.size());
        println!("  Sample tensor range: {:.3} to {:.3}",
            data[0].img.min().double_value(&[]),
            data[0].img.max().double_value(&[]));

        // ── TRAINING PHASE ──────────────────────────────────────────────────
        println!("\n🏋️ Starting Training...");
        let config = TrainingConfig {
            num_epochs,
            learning_rate,
            step_size,
            gamma,
            max_norm: 1.0,
        };

        println!("⚙️ Training Configuration:");
        println!("  Epochs: {}", num_epochs);
        println!("  Learning Rate: {:.6}", learning_rate);
        println!("  Step Size: {}", step_size);
        println!("  Gamma: {}", gamma);

        let start_time = std::time::Instant::now();
        
        // Train the model using train_model.rs functions
        let _stats = train_model(&model, &mut train_loader, &mut val_loader, &vs, config)?;
        
        let training_duration = start_time.elapsed();
        println!("⏱️ Training completed in: {:?}", training_duration);

        // ── COMPREHENSIVE EVALUATION PHASE ──────────────────────────────────
        let class_names = ["cardboard", "glass", "metal", "paper", "plastic", "trash"];

        // Extract losses from training stats  
        let train_losses: Vec<f64> = _stats.training_losses.clone();
        let val_losses: Vec<f64> = _stats.validation_losses.clone();

        // Run complete evaluation pipeline
        run_complete_evaluation(
            &model,
            &mut val_loader,
            device,
            &class_names,
            &train_losses,
            &val_losses,
        )?;

        // ── SAVE MODEL ──────────────────────────────────────────────────────
        let model_name = format!("garbage_classifier_{}_epochs_{}_samples.pt", num_epochs, data.len());
        save_model(&vs, &model_name)?;

        println!("\n🎉 Training and evaluation completed successfully!");
        println!("💾 Model saved as {}", model_name);
        println!("⏱️ Total time: {:?}", training_duration);

        // Ask if user wants to continue
        print!("\nWould you like to run another configuration? (y/n): ");
        std::io::stdout().flush().unwrap();
        let continue_choice = menu::get_user_input();
        if continue_choice.to_lowercase() != "y" {
            break;
        }
    }

    Ok(())
}
