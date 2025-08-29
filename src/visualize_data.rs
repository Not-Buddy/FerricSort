// visualize_data.rs

use crate::Sample;
use tch::{Tensor, Device, Kind};
use rand::seq::SliceRandom;
use anyhow::anyhow;

/// Simple DataLoader for batching samples from preloaded dataset
pub struct DataLoader<'a> {
    data: &'a [Sample],
    batch_size: usize,
    indices: Vec<usize>,
    idx: usize,
    shuffle: bool,
}

impl<'a> DataLoader<'a> {
    pub fn new(data: &'a [Sample], batch_size: usize, shuffle: bool) -> Self {
        let mut indices: Vec<usize> = (0..data.len()).collect();
        if shuffle {
            indices.shuffle(&mut rand::thread_rng());
        }
        DataLoader {
            data,
            batch_size,
            indices,
            idx: 0,
            shuffle,
        }
    }

    /// Returns the next batch as (images tensor, labels tensor)
    pub fn next(&mut self) -> Option<(Tensor, Tensor)> {
        if self.idx >= self.data.len() {
            return None;
        }

        let end = (self.idx + self.batch_size).min(self.data.len());
        let batch_indices = &self.indices[self.idx..end];
        let images = Tensor::stack(
            &batch_indices
                .iter()
                .map(|&i| self.data[i].img.shallow_clone())
                .collect::<Vec<_>>(),
            0,
        );
        let labels = Tensor::from_slice(
            &batch_indices
                .iter()
                .map(|&i| self.data[i].label)
                .collect::<Vec<_>>()
        );
        self.idx = end;
        Some((images, labels))
    }

    /// Resets the DataLoader to start, and reshuffles if desired
    pub fn reset(&mut self) {
        self.idx = 0;
        if self.shuffle {
            self.indices.shuffle(&mut rand::thread_rng());
        }
    }
}

/// Visualizes a batch of images from a DataLoader and saves as a PNG.
/// Each image will be laid out in a grid, similar to PyTorch's make_grid.
/// Requires the `image` crate for manipulation and saving.
pub fn visualize_batch(dl: &mut DataLoader, filename: &str) -> anyhow::Result<()> {
    let (images, _labels) = dl.next().ok_or(anyhow!("No batch available"))?;
    let batch_size = images.size()[0] as usize;
    let height = images.size()[2] as usize;
    let width = images.size()[3] as usize;
    let nrow = 8;
    let ncol = (batch_size + nrow - 1) / nrow;

    let mut imgbuf = image::ImageBuffer::<image::Rgb<u8>, Vec<u8>>::new(
        (width * nrow) as u32,
        (height * ncol) as u32,
    );

    for idx in 0..batch_size {
        let row = idx / nrow;
        let col = idx % nrow;
        let image_tensor = images.get(idx as i64);
        
        // FIXED: Scale to [0,255] before converting to Uint8
        let image_tensor = (image_tensor * 255.0)
            .to_device(Device::Cpu)
            .to_kind(Kind::Uint8)
            .permute(&[1, 2, 0]);

        // FIXED: Use try_from properly
        let flattened = image_tensor.contiguous().view([-1]);
        let data: Vec<u8> = Vec::<u8>::try_from(flattened)
            .expect("Failed to convert tensor to Vec<u8>");

        for y in 0..height {
            for x in 0..width {
                let pixel_idx = (y * width + x) * 3;
                let r = data[pixel_idx];
                let g = data[pixel_idx + 1]; 
                let b = data[pixel_idx + 2];
                imgbuf.put_pixel(
                    (col * width + x) as u32,
                    (row * height + y) as u32,
                    image::Rgb([r, g, b]),
                );
            }
        }
    }

    imgbuf.save(filename)?;
    println!("Batch visualization saved to {}", filename);
    Ok(())
}

