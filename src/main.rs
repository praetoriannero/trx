use eframe::egui;
use egui::{ColorImage, Pos2, TextureHandle, Widget};
// use egui_plot::{Line, Plot};
use num_complex::ComplexFloat;
use rustfft::{FftPlanner, num_complex::Complex};
use smoothed_z_score::{Peak, PeaksDetector, PeaksFilter};
use soapysdr;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::{f64, thread};

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

// const W: usize = 2048;
// const H: usize = 512;

pub struct HeatmapApp {
    buffer: Arc<Mutex<VecDeque<Vec<f64>>>>,
    x_size: i64,
    y_size: i64,
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

struct Signal {
    center_freq: f32,
    bandwidth: f32,
    label: String,
}

struct AnnotatedSpectrogram<'a> {
    spectrogram: &'a TextureHandle,
    signals: Vec<Signal>,
}

impl eframe::App for HeatmapApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let mut img = ColorImage::new(
            [self.x_size as usize, self.y_size as usize],
            vec![egui::Color32::BLACK; self.x_size as usize * self.y_size as usize],
        );
        let buff = self.buffer.lock().unwrap();
        for y in 0..self.y_size {
            for x in 0..self.x_size {
                let row = buff.get(y as usize);
                if row.is_none() {
                    return;
                }
                let pixel = row.unwrap().get(x as usize);
                if pixel.is_none() {
                    return;
                }
                let v = pixel.unwrap();
                let intensity = (sigmoid(*v) * 255.0) as u8;
                img[(x as usize, y as usize)] = egui::Color32::from_rgb(intensity, intensity, 0);
            }
        }
        let font_family = egui::FontFamily::Monospace;
        let font_id = egui::FontId::new(11.0, font_family);
        let text_color = egui::Color32::from_rgb(255, 0, 0);
        let tex = ctx.load_texture("spectrogram", img, Default::default());
        let sized_tex = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
        let spec_tex = egui::Image::new(sized_tex);
        let stroke = egui::Stroke::new(1.0, egui::Color32::from_rgb(255, 0, 0));
        // Start draw
        egui::CentralPanel::default().show(ctx, |ui| {
            let response = ui.add(spec_tex);
            let spec_rect: egui::Rect = response.rect;
            let mut points = vec![spec_rect.left_top(), spec_rect.right_bottom()];
            ui.painter().line(points, stroke);
            points = vec![spec_rect.left_bottom(), spec_rect.right_top()];
            ui.painter().line(points, stroke);
            ui.painter().text(
                spec_rect.center(),
                egui::Align2::CENTER_CENTER,
                "test",
                font_id,
                text_color,
            );
        });

        ctx.request_repaint();
    }
}

struct SdrConfig {
    center_freq: f64,
    sample_rate: f64,
    receive_gain: f64,
    timeout_us: i64,
    sample_len: i64,
    down_sample_ratio: usize,
    fft_bins: Vec<f64>,
}

struct SpectralDensity {
    spectrum: Vec<f64>,
    samples: i64,
}

impl SpectralDensity {
    fn average_power(&self) -> Vec<f64> {
        self.spectrum
            .iter()
            .map(|f| f / self.samples as f64)
            .collect()
    }
}

fn main() -> Result<(), eframe::Error> {
    let center_freq = 103_000_000.0;
    let sample_rate = 20_000_000.0;
    let receive_gain = 0.0;
    let timeout_us = 1_000_000;
    let sample_len = 4096;
    let x_size = 2048;
    let y_size = 720;

    let down_sample_ratio = (sample_len / x_size) as usize;
    let fft_bins = fft_freqs(sample_len, 1.0 / sample_rate);
    let config = SdrConfig {
        center_freq: center_freq,
        sample_rate: sample_rate,
        receive_gain: receive_gain,
        timeout_us: timeout_us,
        sample_len: sample_len as i64,
        down_sample_ratio: down_sample_ratio,
        fft_bins: fft_bins,
    };

    let rx_samples: Arc<Mutex<Vec<f64>>> =
        Arc::new(Mutex::new(Vec::with_capacity(config.sample_len as usize)));
    let rx_samples_clone = rx_samples.clone();
    let heatmap_deque: Arc<Mutex<VecDeque<Vec<f64>>>> =
        Arc::new(Mutex::new(VecDeque::with_capacity(y_size)));
    {
        let mut buff = heatmap_deque.lock().unwrap();
        for _ in 0..y_size {
            let row: Vec<f64> = Vec::with_capacity(config.sample_len as usize);
            buff.push_front(row);
        }
    }
    let heatmap_clone = heatmap_deque.clone();
    let detected_signals: Arc<Mutex<Vec<Signal>>> =
        Arc::new(Mutex::new(Vec::with_capacity(config.sample_len as usize)));

    // Start SDR thread
    thread::spawn(move || {
        println!("Spawning SDR thread");
        let sdr = soapysdr::Device::new("driver=hackrf").expect("HackRF not found");
        sdr.set_frequency(soapysdr::Direction::Rx, 0, config.center_freq, ())
            .unwrap();
        sdr.set_sample_rate(soapysdr::Direction::Rx, 0, config.sample_rate)
            .unwrap();
        sdr.set_gain(soapysdr::Direction::Rx, 0, config.receive_gain)
            .unwrap();

        let mut spectral_sum: Vec<f64> = Vec::with_capacity(sample_len as usize);
        for _ in 0..sample_len {
            spectral_sum.push(0.0);
        }
        let mut spectral_density = SpectralDensity {
            spectrum: spectral_sum,
            samples: 0,
        };
        let mut rx = sdr.rx_stream::<Complex<f32>>(&[0]).unwrap();
        rx.activate(None).unwrap();

        let mut planner: FftPlanner<f32> = FftPlanner::new();
        let fft = planner.plan_fft_forward(sample_len as usize);
        let mut buff = vec![
            Complex {
                re: 0.0f32,
                im: 0.0f32
            };
            sample_len as usize
        ];
        loop {
            spectral_density.samples += 1;
            let _ = rx.read(&mut [&mut buff], config.timeout_us).unwrap_or(0);
            let mut data = rx_samples_clone.lock().unwrap();
            fft.process(&mut buff);
            data.clear();
            let abs_buff: Vec<f32> = buff.iter().map(|s| s.abs()).collect();
            let mut ordered_buff: Vec<f32> = Vec::new();
            for f in abs_buff.len() / 2..abs_buff.len() {
                ordered_buff.push(abs_buff[f]);
            }
            for f in 0..abs_buff.len() / 2 {
                ordered_buff.push(abs_buff[f]);
            }
            let abs_vec: Vec<f32> = buff.iter().map(|c| c.abs() as f32).collect();
            // println!("{} {}", abs_vec.len(), spectral_density.spectrum.len());
            for i in 0..abs_vec.len() {
                spectral_density.spectrum[i] += abs_vec[i] as f64;
            }
            let avg_power = spectral_density.average_power();
            let peaks: Vec<_> = avg_power
                .clone()
                .into_iter()
                .enumerate()
                .peaks(PeaksDetector::new(30, 5.0, 0.0), |e| e.1 as f64)
                .map(|((i, _), p)| (i, p))
                .collect();
            let mut peaks_actual: Vec<f64> = Vec::new();
            let mut last_idx = 0usize;
            let mut indices: Vec<f64> = Vec::new();
            for (index, peak_type) in peaks.iter() {
                if last_idx == 0 {
                    last_idx = *index;
                    continue;
                }
                if *index == last_idx + 1 {
                    indices.push(avg_power[peaks[*index].0]);
                } else {
                    indices.clear();
                }
                println!("{:?} {:?}", index, peak_type);
            }
            // println!("{:?}", peaks);
            for i in 0..x_size {
                let j = i * down_sample_ratio as u64;
                let k = (i + 1) * down_sample_ratio as u64;
                let sum: f32 = ordered_buff[j as usize..k as usize]
                    .iter()
                    .map(|x| *x)
                    .sum();
                let mean = sum / down_sample_ratio as f32;
                data.push(mean as f64);
            }
            // likely will need to run a notched filter over baseband
            // and capture off of center
            // data[0] = 0.0; // TODO: figure out how to handle the DC spike
            let hm = heatmap_clone.lock();
            match hm {
                Ok(mut hm_result) => {
                    hm_result.pop_back();
                    let row = data
                        .iter()
                        .map(|x| 10.0 * x.abs().log(10.0) as f64)
                        .collect();
                    hm_result.push_front(row);
                }
                Err(_) => continue,
            };
            // for i in 0..y_size {
            //     for j in 0..x_size {
            //         continue;
            //     }
            // }
        }
    });

    // let app = VisualizerApp { samples: samples };
    let app = HeatmapApp {
        buffer: heatmap_deque,
        x_size: x_size as i64,
        y_size: y_size as i64,
    };
    let viewport = egui::ViewportBuilder {
        inner_size: Some(egui::Vec2::new(x_size as f32 + 16.0, y_size as f32 + 16.0)),
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
