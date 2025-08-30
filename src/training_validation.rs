// training_validation.rs

use crate::relu::CNN; 
use crate::visualize_data::DataLoader;
use tch::{nn::ModuleT, Device, Kind};
use std::fs::File;
use std::io::Write;

/// Structure to hold evaluation results
pub struct EvaluationResults {
    pub predictions: Vec<i64>,
    pub true_labels: Vec<i64>,
    pub probabilities: Vec<Vec<f64>>,
}

/// Save training and validation losses to CSV (equivalent to matplotlib plotting data)
pub fn save_loss_csv(
    train_losses: &[f64],
    val_losses: &[f64],
    filename: &str,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    writeln!(file, "epoch,train_loss,val_loss")?;
    
    for (epoch, (&train_loss, &val_loss)) in train_losses.iter().zip(val_losses.iter()).enumerate() {
        writeln!(file, "{},{:.6},{:.6}", epoch + 1, train_loss, val_loss)?;
    }
    
    println!("Loss data saved to {}", filename);
    Ok(())
}

/// Evaluate model (equivalent to Python @torch.no_grad() eval function)
pub fn eval_model(
    model: &CNN,
    loader: &mut DataLoader,
    device: Device,
) -> anyhow::Result<EvaluationResults> {
    let mut preds_probs = Vec::new();
    let mut trues = Vec::new();
    
    loader.reset();
    
    tch::no_grad(|| {
        while let Some((images, labels)) = loader.next() {
            let images = images.to_device(device);
            let labels = labels.to_device(device);
            
            // Forward pass: Get logits (raw output from the model)
            let outputs = model.forward_t(&images, false); // model.eval() equivalent
            
            // Convert logits to probabilities (softmax for multi-class classification)
            let probs = outputs.softmax(1, Kind::Float); // torch.softmax(outputs, dim=1)
            
            // Store predicted probabilities on CPU
            let probs_cpu = probs.to_device(Device::Cpu);
            let labels_cpu = labels.to_device(Device::Cpu);
            
            // Extract probabilities and labels for each sample
            for i in 0..probs_cpu.size()[0] {
                // Get probability vector for this sample
                let mut prob_vec = Vec::new();
                for j in 0..probs_cpu.size()[1] {
                    prob_vec.push(probs_cpu.double_value(&[i, j]));
                }
                preds_probs.push(prob_vec);
                trues.push(labels_cpu.int64_value(&[i]));
            }
        }
    });
    
    // Get predicted classes by taking argmax of probabilities
    let predictions: Vec<i64> = preds_probs
        .iter()
        .map(|probs| {
            probs
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx as i64)
                .unwrap_or(0)
        })
        .collect();
    
    Ok(EvaluationResults {
        predictions,
        true_labels: trues,
        probabilities: preds_probs,
    })
}

/// Calculate classification report (equivalent to sklearn.metrics.classification_report)
pub fn print_classification_report(
    actual: &[i64],
    pred: &[i64],
    class_names: &[&str],
) {
    println!("\nClassification Report:");
    println!("{:>12} {:>10} {:>10} {:>10} {:>10}", "Class", "Precision", "Recall", "F1-Score", "Support");
    println!("{}", "-".repeat(60));
    
    let mut total_samples = 0;
    let mut weighted_precision = 0.0;
    let mut weighted_recall = 0.0;
    let mut weighted_f1 = 0.0;
    
    for (class_idx, &class_name) in class_names.iter().enumerate() {
        let class_idx = class_idx as i64;
        
        // Calculate TP, FP, FN for this class
        let mut tp = 0;
        let mut fp = 0;
        let mut fn_count = 0;
        let mut support = 0;
        
        for (&p, &t) in pred.iter().zip(actual.iter()) {
            if t == class_idx {
                support += 1;
                if p == class_idx {
                    tp += 1;
                } else {
                    fn_count += 1;
                }
            } else if p == class_idx {
                fp += 1;
            }
        }
        
        // Calculate metrics
        let precision = if tp + fp > 0 { tp as f64 / (tp + fp) as f64 } else { 0.0 };
        let recall = if tp + fn_count > 0 { tp as f64 / (tp + fn_count) as f64 } else { 0.0 };
        let f1 = if precision + recall > 0.0 {
            2.0 * precision * recall / (precision + recall)
        } else {
            0.0
        };
        
        println!("{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
            class_name, precision, recall, f1, support);
        
        total_samples += support;
        weighted_precision += precision * support as f64;
        weighted_recall += recall * support as f64;
        weighted_f1 += f1 * support as f64;
    }
    
    // Calculate weighted averages
    if total_samples > 0 {
        weighted_precision /= total_samples as f64;
        weighted_recall /= total_samples as f64;
        weighted_f1 /= total_samples as f64;
    }
    
    println!("{}", "-".repeat(60));
    println!("{:>12} {:>10.2} {:>10.2} {:>10.2} {:>10}",
        "Weighted Avg", weighted_precision, weighted_recall, weighted_f1, total_samples);
}

/// Calculate confusion matrix (equivalent to sklearn.metrics.confusion_matrix)
pub fn confusion_matrix(actual: &[i64], pred: &[i64], num_classes: usize) -> Vec<Vec<i64>> {
    let mut matrix = vec![vec![0; num_classes]; num_classes];
    
    for (&p, &t) in pred.iter().zip(actual.iter()) {
        if t >= 0 && t < num_classes as i64 && p >= 0 && p < num_classes as i64 {
            matrix[t as usize][p as usize] += 1;
        }
    }
    
    matrix
}

/// Print confusion matrix with class names
pub fn print_confusion_matrix(matrix: &[Vec<i64>], class_names: &[&str]) {
    println!("\nConfusion Matrix:");
    println!("Predicted ->");
    
    // Header
    print!("Actual\\    ");
    for name in class_names {
        print!("{:>10} ", name);
    }
    println!();
    
    // Matrix rows
    for (i, row) in matrix.iter().enumerate() {
        print!("{:>10} ", class_names[i]);
        for &val in row {
            print!("{:>10} ", val);
        }
        println!();
    }
    println!();
}

/// Save confusion matrix to CSV for plotting with seaborn/matplotlib
pub fn save_confusion_matrix_csv(
    matrix: &[Vec<i64>],
    class_names: &[&str],
    filename: &str,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header with class names
    write!(file, "actual\\predicted")?;
    for name in class_names {
        write!(file, ",{}", name)?;
    }
    writeln!(file)?;
    
    // Write matrix rows
    for (i, row) in matrix.iter().enumerate() {
        write!(file, "{}", class_names[i])?;
        for &val in row {
            write!(file, ",{}", val)?;
        }
        writeln!(file)?;
    }
    
    println!("Confusion matrix saved to {}", filename);
    Ok(())
}

/// Save evaluation results to CSV (equivalent to creating a pandas DataFrame)
pub fn save_evaluation_results_csv(
    results: &EvaluationResults,
    filename: &str,
) -> std::io::Result<()> {
    let mut file = File::create(filename)?;
    
    // Write header
    writeln!(file, "actual,pred,probability_0,probability_1,probability_2,probability_3,probability_4,probability_5")?;
    
    // Write data rows
    for (i, (&actual, &pred)) in results.true_labels.iter().zip(results.predictions.iter()).enumerate() {
        write!(file, "{},{}", actual, pred)?;
        for &prob in &results.probabilities[i] {
            write!(file, ",{:.6}", prob)?;
        }
        writeln!(file)?;
    }
    
    println!("Evaluation results saved to {}", filename);
    Ok(())
}

/// Generate Python script for plotting (equivalent to matplotlib/seaborn plotting)
pub fn generate_plot_script(
    loss_csv: &str,
    confusion_csv: &str,
    script_name: &str,
) -> std::io::Result<()> {
    let mut file = File::create(script_name)?;
    
    writeln!(file, "import matplotlib.pyplot as plt")?;
    writeln!(file, "import pandas as pd")?;
    writeln!(file, "import seaborn as sns")?;
    writeln!(file, "import numpy as np")?;
    writeln!(file)?;
    
    // Loss plotting (equivalent to your matplotlib code)
    writeln!(file, "# Choose a style for the plot")?;
    writeln!(file, "plt.style.use('default')")?;
    writeln!(file)?;
    writeln!(file, "# Load loss data")?;
    writeln!(file, "loss_data = pd.read_csv('{}')", loss_csv)?;
    writeln!(file, "epochs = loss_data['epoch']")?;
    writeln!(file, "training_loss = loss_data['train_loss']")?;
    writeln!(file, "validation_loss = loss_data['val_loss']")?;
    writeln!(file)?;
    writeln!(file, "# Create figure")?;
    writeln!(file, "plt.figure(figsize=(10, 7))")?;
    writeln!(file)?;
    writeln!(file, "# Plot training and validation loss")?;
    writeln!(file, "plt.plot(epochs, training_loss, label='Training Loss',")?;
    writeln!(file, "         color='#ffda06', linestyle='-', linewidth=2.2, alpha=1)")?;
    writeln!(file)?;
    writeln!(file, "plt.plot(epochs, validation_loss, label='Validation Loss',")?;
    writeln!(file, "         color='red', linestyle='--', linewidth=2, alpha=1)")?;
    writeln!(file)?;
    writeln!(file, "# Labels and title")?;
    writeln!(file, "plt.xlabel('Epochs', fontsize=14, fontweight='bold', color='#333333')")?;
    writeln!(file, "plt.ylabel('Loss', fontsize=14, fontweight='bold', color='#333333')")?;
    writeln!(file, "plt.title('Training and Validation Loss', fontsize=17, fontweight='bold', color='#222222')")?;
    writeln!(file)?;
    writeln!(file, "plt.grid(True, linestyle='--', linewidth=0.7, alpha=0.6)")?;
    writeln!(file, "plt.legend(loc='upper right', fontsize=13, frameon=False)")?;
    writeln!(file)?;
    writeln!(file, "# Adjust layout for a clean look")?;
    writeln!(file, "plt.tight_layout()")?;
    writeln!(file, "plt.savefig('training_validation_loss.png', dpi=300, bbox_inches='tight')")?;
    writeln!(file, "plt.show()")?;
    writeln!(file)?;
    
    // Confusion matrix plotting (equivalent to seaborn heatmap)
    writeln!(file, "# Load confusion matrix data")?;
    writeln!(file, "conf_data = pd.read_csv('{}', index_col=0)", confusion_csv)?;
    writeln!(file)?;
    writeln!(file, "# Create a heatmap to visualize the confusion matrix")?;
    writeln!(file, "plt.figure(figsize=(10, 7))  # Set figure size")?;
    writeln!(file)?;
    writeln!(file, "# Plot the confusion matrix with annotations and class names")?;
    writeln!(file, "sns.heatmap(conf_data, annot=True, fmt='d', cmap='viridis')")?;
    writeln!(file)?;
    writeln!(file, "# Add labels and title to the plot")?;
    writeln!(file, "plt.title('Confusion Matrix')")?;
    writeln!(file, "plt.xlabel('Predicted Labels')")?;
    writeln!(file, "plt.ylabel('True Labels')")?;
    writeln!(file)?;
    writeln!(file, "plt.tight_layout()")?;
    writeln!(file, "plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')")?;
    writeln!(file, "# Show the plot")?;
    writeln!(file, "plt.show()")?;
    
    println!("Python plotting script saved to {}", script_name);
    println!("Run with: python {}", script_name);
    Ok(())
}

/// Complete evaluation pipeline (combines all functions)
pub fn run_complete_evaluation(
    model: &CNN,
    val_loader: &mut DataLoader,
    device: Device,
    class_names: &[&str],
    train_losses: &[f64],
    val_losses: &[f64],
) -> anyhow::Result<()> {
    println!("🔍 Running complete evaluation pipeline...");
    
    // 1. Save loss data for plotting
    save_loss_csv(train_losses, val_losses, "training_losses.csv")?;
    
    // 2. Evaluate model (equivalent to val_res = eval(val_loader))
    let val_res = eval_model(model, val_loader, device)?;
    
    // 3. Calculate evaluation metrics (equivalent to classification_report)
    print_classification_report(&val_res.true_labels, &val_res.predictions, class_names);
    
    // 4. Calculate confusion matrix
    let confusion_mat = confusion_matrix(&val_res.true_labels, &val_res.predictions, class_names.len());
    print_confusion_matrix(&confusion_mat, class_names);
    
    // 5. Save data for plotting
    save_confusion_matrix_csv(&confusion_mat, class_names, "confusion_matrix.csv")?;
    save_evaluation_results_csv(&val_res, "evaluation_results.csv")?;
    
    // 6. Generate Python plotting script
    generate_plot_script("training_losses.csv", "confusion_matrix.csv", "plot_results.py")?;
    
    println!("✅ Complete evaluation finished!");
    println!("📊 Files generated:");
    println!("   - training_losses.csv");
    println!("   - confusion_matrix.csv");  
    println!("   - evaluation_results.csv");
    println!("   - plot_results.py");
    println!("\n📈 To generate plots, run: python plot_results.py");
    
    Ok(())
}
