// menu.rs
use std::{fs, path::Path};
use image::{imageops, DynamicImage};
use imageproc::geometric_transformations::{rotate_about_center, Interpolation};
use rand::{seq::SliceRandom, Rng};
use tch::{Tensor};
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