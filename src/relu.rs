// relu.rs

use tch::nn::{self, ModuleT};
use tch::Tensor;

// CNN model mimicking the Python nn.Module class
#[derive(Debug)]
pub struct CNN {
    features: nn::SequentialT,
    gap: nn::FuncT<'static>, 
    classifier: nn::SequentialT,
}

impl CNN {
    pub fn new(vs: &nn::Path, num_classes: i64, dropout_rate: f64) -> CNN {
        let features = nn::seq_t()
            // Feature extraction layers (Conv + BN + ReLU + Pooling)
            .add(nn::conv2d(vs, 3, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 32, 32, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 32, Default::default()))
            .add_fn(|xs| xs.relu())
            // Downsampling
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            .add(nn::conv2d(vs, 32, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 64, 64, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 64, Default::default()))
            .add_fn(|xs| xs.relu())
            // Downsampling
            .add_fn(|xs| xs.max_pool2d_default(2))
            
            .add(nn::conv2d(vs, 64, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add(nn::conv2d(vs, 128, 128, 3, nn::ConvConfig { padding: 1, ..Default::default() }))
            .add(nn::batch_norm2d(vs, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            // Downsampling
            .add_fn(|xs| xs.max_pool2d_default(2));

        // Global average pooling
        let gap = nn::func_t(|xs, _train| xs.adaptive_avg_pool2d(&[1, 1]));

        // Fully connected classifier
        let classifier = nn::seq_t()
            .add_fn(|xs| xs.flatten(1, -1))
            .add(nn::linear(vs, 128, 128, Default::default()))
            .add_fn(|xs| xs.relu())
            .add_fn_t(move |xs, train| xs.dropout(dropout_rate, train))
            .add(nn::linear(vs, 128, num_classes, Default::default()));

        CNN { features, gap, classifier }
    }
}

impl nn::ModuleT for CNN {
    fn forward_t(&self, xs: &Tensor, train: bool) -> Tensor {
        let x = self.features.forward_t(xs, train);   // extract features
        let x = self.gap.forward_t(&x, train);        // global average pooling
        self.classifier.forward_t(&x, train)          // classification head
    }
}
