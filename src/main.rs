use eframe::egui;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

mod app;
mod sdr;
mod signal;

fn main() -> Result<(), eframe::Error> {
    let config = sdr::SdrConfig::default();
    // let rx_samples: Arc<Mutex<Vec<f64>>> =
    //     Arc::new(Mutex::new(Vec::with_capacity(config.sample_len as usize)));
    // let rx_samples_clone = rx_samples.clone();
    let heatmap_deque: Arc<Mutex<VecDeque<Vec<f64>>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(config.y_size)));
    {
        let mut buff = heatmap_deque.lock().unwrap();
        for _ in 0..config.y_size {
            let row: Vec<f64> = Vec::with_capacity(config.sample_len as usize);
            buff.push_front(row);
        }
    }
    let heatmap_clone = heatmap_deque.clone();
    sdr::spawn_listener(config.clone(), heatmap_clone);
    // let app = VisualizerApp { samples: samples };
    let app = app::HeatmapApp {
        buffer: heatmap_deque,
        x_size: config.x_size as i64,
        y_size: config.y_size as i64,
    };
    let viewport = egui::ViewportBuilder {
        inner_size: Some(egui::Vec2::new(
            config.x_size as f32 + 16.0,
            config.y_size as f32 + 16.0,
        )),
        ..Default::default()
    };
    let options = eframe::NativeOptions {
        viewport: viewport,
        ..Default::default()
    };
    eframe::run_native(
        "Spectrum Visualizer",
        options,
        Box::new(|_| Ok(Box::new(app))),
    )
}

// fn rolling_average(data: &[f64], window_size: usize) -> Vec<f64> {
//     if window_size == 0 {
//         return Vec::new(); // Handle zero window size
//     }
//
//     let mut averages = Vec::with_capacity(data.len());
//     for i in 0..data.len() {
//         let start_index = if i >= window_size - 1 {
//             i - (window_size - 1)
//         } else {
//             0
//         };
//         let end_index = i + 1; // Exclusive end
//
//         let window = &data[start_index..end_index];
//         let sum: f64 = window.iter().sum();
//         let average = sum / (window.len() as f64);
//         averages.push(average);
//     }
//     averages
// }

// fn normalize_vec(vector: &mut Vec<f64>) {
//     let norm = vector
//         .iter()
//         .copied()
//         .filter(|x| !x.is_nan())
//         .max_by(|a, b| a.partial_cmp(b).unwrap())
//         .unwrap();
//     for i in 0..vector.len() {
//         vector[i] = vector[i] / norm;
//     }
// }

// struct VisualizerApp {
//     samples: Arc<Mutex<Vec<f64>>>,
// }

// impl Default for HeatmapApp {
//     fn default() -> Self {
//         let buffer: Arc<Mutex<VecDeque<Vec<f64>>>> =
//             Arc::new(Mutex::new(VecDeque::with_capacity(H)));
//         let mut buff = buffer.lock().unwrap();
//         for _ in 0..W {
//             let row: Vec<f64> = Vec::with_capacity(W);
//             buff.push_front(row);
//         }
//         Self {
//             tex: None,
//             buffer: buffer.clone(),
//         }
//     }
// }
