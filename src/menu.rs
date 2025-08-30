// menu.rs
use std::io::{self, Write};
use std::{fs, path::Path};
use image::{imageops, DynamicImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rand::{seq::SliceRandom, Rng};
use tch::{Device, Tensor};
use rayon::prelude::*;
use std::sync::atomic::{AtomicUsize, Ordering};

/// Sample structure for image and label data
#[derive(Debug)]
pub struct Sample {
    pub img: Tensor,
    pub label: i64,
}

/// Recursively loads every image under `root`, infers its label from the
/// parent folder name, performs data-augmentation, and returns a Vec<Sample>.
pub fn load_dataset(root: &str) -> Vec<Sample> {
    // ── 1. collect (path, label) pairs ───────────────────────────────────────
    let mut entries = Vec::new();
    let root = Path::new(root);
    for (label_idx, dir) in fs::read_dir(root)
        .expect("cannot read root directory")
        .filter_map(Result::ok)
        .filter(|d| d.path().is_dir())
        .enumerate()
    {
        let label = label_idx as i64;
        for file in fs::read_dir(dir.path())
            .expect("cannot read class directory")
            .filter_map(Result::ok)
            .filter(|d| d.path().is_file())
        {
            entries.push((file.path(), label));
        }
    }

    // shuffle entries
    let mut rng = rand::thread_rng();
    entries.shuffle(&mut rng);
    println!("Processing {} images in parallel...", entries.len());
    
    // ── 2. Parallel processing with Rayon and atomic counter ────────────────
    let counter = AtomicUsize::new(0);
    let results: Vec<Sample> = entries
        .par_iter()
        .filter_map(|(path, label)| {
            let count = counter.fetch_add(1, Ordering::Relaxed);
            if count % 500 == 0 {
                println!("Processed {} images...", count);
            }

            // Create a thread-local RNG for each worker
            let mut rng = rand::thread_rng();
            // Image loading and processing
            let mut img = image::open(&path).ok()?.into_rgb8();
            if rng.gen_bool(0.5) { imageops::flip_horizontal_in_place(&mut img); }
            if rng.gen_bool(0.5) { imageops::flip_vertical_in_place(&mut img); }
            let angle = rng.gen_range(-1.0_f32..1.0_f32).to_radians();
            let rotated = rotate_about_center(&img, angle, Interpolation::Bilinear, image::Rgb([0, 0, 0]));
            let resized = DynamicImage::ImageRgb8(rotated)
                .resize_exact(200, 200, imageops::FilterType::Nearest)
                .to_rgb8();
                
            let pixels = resized.as_raw();
            let mut chw_data = vec![0.0f32; 3 * 200 * 200];
            for i in 0..pixels.len() {
                let c = i % 3;
                let hw = i / 3;
                chw_data[c * 200 * 200 + hw] = pixels[i] as f32 / 255.0;
            }
            let tensor = Tensor::from_slice(&chw_data).view([3, 200, 200]);
            Some(Sample { img: tensor, label: *label })
        })
        .collect();
    println!("Completed processing {} images", results.len());
    results
}

pub fn display_menu() {
    println!("\n🗂️ Garbage Classification Training Menu");
    println!("{}", "=".repeat(50));
    println!("1. 🚀 Quick Test (500 samples, 2 epochs, CPU)");
    println!("2. 🔬 Medium Test (2000 samples, 5 epochs, CPU/GPU)");
    println!("3. 📊 Full Training (All samples, 100 epochs, GPU recommended)");
    println!("4. 🎯 Custom Configuration");
    println!("5. ❌ Exit");
    println!("{}", "=".repeat(50));
    print!("Enter your choice (1-5): ");
}

pub fn get_user_input() -> String {
    let mut input = String::new();
    io::stdin().read_line(&mut input).expect("Failed to read input");
    input.trim().to_string()
}

pub fn get_training_config(choice: &str) -> Option<(usize, i64, f64, i64, f64, i64, Device)> {
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
            println!("📊 Full Training Configuration Selected");
            Some((usize::MAX, 100, 1e-4, 5, 0.85, 32, Device::cuda_if_available()))
        },
        "4" => {
            println!("🎯 Custom Configuration");
            
            print!("Enter max samples (or 0 for all): ");
            io::stdout().flush().unwrap();
            let max_samples = get_user_input().parse::<usize>().unwrap_or(1000);
            let max_samples = if max_samples == 0 { usize::MAX } else { max_samples };
            
            print!("Enter number of epochs: ");
            io::stdout().flush().unwrap();
            let epochs = get_user_input().parse::<i64>().unwrap_or(5);
            
            print!("Enter learning rate (e.g., 0.001): ");
            io::stdout().flush().unwrap();
            let lr = get_user_input().parse::<f64>().unwrap_or(1e-3);
            
            print!("Enter batch size: ");
            io::stdout().flush().unwrap();
            let batch_size = get_user_input().parse::<i64>().unwrap_or(16);
            
            print!("Use GPU if available? (y/n): ");
            io::stdout().flush().unwrap();
            let use_gpu = get_user_input().to_lowercase() == "y";
            let device = if use_gpu { Device::cuda_if_available() } else { Device::Cpu };
            
            Some((max_samples, epochs, lr, 2, 0.85, batch_size, device))
        },
        _ => None,
    }
}
