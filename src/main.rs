use eframe::egui;
use egui::{ColorImage, TextureHandle};
use egui_plot::{Line, Plot};
use num_complex::ComplexFloat;
use rustfft::{FftPlanner, num_complex::Complex};
use soapysdr;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

const W: usize = 2048;
const H: usize = 512;

pub struct HeatmapApp {
    tex: Option<TextureHandle>,
    buffer: Arc<Mutex<VecDeque<Vec<f64>>>>,
}

fn fft_freqs(samples: u64, duration: f64) -> Vec<f64> {
    let mut freqs: Vec<f64> = Vec::new();
    if samples % 2 == 0 {
        let len = (samples as i32 / 2) - 1;
        for freq in 0..=len {
            let f = freq as f64 / (duration * samples as f64);
            freqs.push(f);
        }
        for freq in (-len - 1)..0 {
            let f = freq as f64 / (duration * samples as f64);
            freqs.push(f);
        }
    } else {
        let len = (samples - 1) as i32 / 2;
        for freq in 0..=len {
            let f = freq as f64 / (duration * samples as f64);
            freqs.push(f);
        }
        for freq in -len..0 {
            let f = freq as f64 / (duration * samples as f64);
            freqs.push(f);
        }
    };
    freqs
}

impl eframe::App for HeatmapApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let mut img = ColorImage::new([W, H], vec![egui::Color32::BLACK; W * H]);
        let buff = self.buffer.lock().unwrap();
        for y in 0..H {
            for x in 0..W {
                let v = buff[y][x];
                let intensity = (sigmoid(v) * 255.0) as u8;
                img[(x, y)] = egui::Color32::from_rgb(intensity, intensity, 0);
            }
        }

        let tex = self.tex.get_or_insert_with(|| {
            ctx.load_texture("", img.clone(), egui::TextureOptions::NEAREST)
        });
        let sized_tex = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
        tex.set(img, egui::TextureOptions::NEAREST);
        let slider_range = 0..=1000;
        let mut initial_value = 10;
        // let line = Line::new("Time Series", buff[0].iter().collect());
        // let line = Line::new(buff[0].iter().collect());
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(egui::Image::new(sized_tex));
            ui.add(egui::Slider::new(&mut initial_value, slider_range));
            // ui.add(Plot::new("Time Series").show(ui, |plot_ui| {
            //     plot_ui.line(line);
            // }));
        });

        ctx.request_repaint();
    }
}

fn main() -> Result<(), eframe::Error> {
    let center_freq = 104_250_000.0;
    let sample_rate = 20_000_000.0;
    let receive_gain = 0.0;
    let timeout_us = 1_000_000;
    let sample_len = 4096;
    let size = 512;

    let samples: Arc<Mutex<Vec<f64>>> = Arc::new(Mutex::new(Vec::with_capacity(sample_len)));
    let samples_clone = samples.clone();
    let heatmap_deque: Arc<Mutex<VecDeque<Vec<f64>>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(size)));
    let heatmap_clone = heatmap_deque.clone();
    {
        let mut buff = heatmap_deque.lock().unwrap();
        for _ in 0..W {
            let row: Vec<f64> = Vec::with_capacity(W);
            buff.push_front(row);
        }
    }
    let freqbins = fft_freqs(7u64, 0.1);
    println!("{:?}", freqbins);
    thread::spawn(move || {
        println!("Spawning SDR thread");
        let sdr = soapysdr::Device::new("driver=hackrf").expect("HackRF not found");
        sdr.set_frequency(soapysdr::Direction::Rx, 0, center_freq, ())
            .unwrap();
        sdr.set_sample_rate(soapysdr::Direction::Rx, 0, sample_rate)
            .unwrap();
        sdr.set_gain(soapysdr::Direction::Rx, 0, receive_gain)
            .unwrap();

        let mut rx = sdr.rx_stream::<Complex<f32>>(&[0]).unwrap();
        rx.activate(None).unwrap();

        let mut planner: FftPlanner<f32> = FftPlanner::new();
        let fft = planner.plan_fft_forward(sample_len);
        let mut buff = vec![
            Complex {
                re: 0.0f32,
                im: 0.0f32
            };
            sample_len
        ];
        loop {
            let n = rx.read(&mut [&mut buff], timeout_us).unwrap_or(0);
            let mut data = samples_clone.lock().unwrap();
            fft.process(&mut buff);
            data.clear();
            for i in 0..n {
                data.push(buff[i].abs() as f64);
            }
            // likely will need to run a notched filter over baseband
            // and capture off of center
            data[0] = 0.0; // TODO: figure out how to handle the DC spike
            let hm = heatmap_clone.lock();
            match hm {
                Ok(mut hm_result) => {
                    hm_result.pop_back();
                    let row = data
                        .iter()
                        // .map(|x| x.abs() as f64)
                        .map(|x| 10.0 * x.abs().log(10.0) as f64)
                        .collect();
                    hm_result.push_front(row);
                }
                Err(_) => continue,
            };
        }
    });

    // let app = VisualizerApp { samples: samples };
    let app = HeatmapApp {
        buffer: heatmap_deque,
        tex: None,
    };
    let options = eframe::NativeOptions {
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
