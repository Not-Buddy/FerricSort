// relu.rs

use tch::nn::{self};
use tch::Tensor;

// Enhanced CNN model with deeper architecture
#[derive(Debug)]
pub struct CNN {
    features: nn::SequentialT,
    gap: nn::FuncT<'static>,
    classifier: nn::SequentialT,
}

impl CNN {
    pub fn new(vs: &nn::Path, num_classes: i64, dropout_rate: f64) -> CNN {
        let features = nn::seq_t()
            // Block 1: 32 filters
            .add(nn::conv2d(vs, 3, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            // Block 2: 64 filters
            .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 64, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            // Block 3: 128 filters
            .add(nn::conv2d(vs, 64, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 128, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            // NEW Block 4: 256 filters - This is the key addition
            .add(nn::conv2d(vs, 128, 256, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 256, 256, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 256, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            // NEW Block 5: 512 filters - Another key addition
            .add(nn::conv2d(vs, 256, 512, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 512, 512, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn(|xs| xs.adaptive_avg_pool2d(&[1, 1])); // Built-in GAP

        // Global average pooling (simplified since we do it in features now)
        let gap = nn::func_t(|xs, _train| xs.view([-1, 512])); // Just reshape

        // Enhanced fully connected classifier
        let classifier = nn::seq_t()
            .add(nn::linear(vs, 512, 512, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(move |xs, train| xs.dropout(dropout_rate, train))
            .add(nn::linear(vs, 512, 256, Default::default()))  // Additional layer
            .add_fn(|xs| xs.relu())
            .add_fn_t(move |xs, train| xs.dropout(dropout_rate * 0.5, train))
            .add(nn::linear(vs, 256, num_classes, Default::default()));

        CNN { features, gap, classifier }
    }
}

impl nn::ModuleT for CNN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x = self.features.forward_t(xs, train);
        let x = self.gap.forward_t(&x, train);
        self.classifier.forward_t(&x, train)
    }
}
