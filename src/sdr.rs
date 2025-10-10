use num_complex::ComplexFloat;
use rustfft::{FftPlanner, num_complex::Complex};
// use smoothed_z_score::{PeaksDetector, PeaksFilter};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};
use std::thread;

use crate::signal;

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

#[derive(Clone)]
pub struct SdrConfig {
    pub center_freq: f64,
    pub sample_rate: f64,
    pub receive_gain: f64,
    pub timeout_us: i64,
    pub sample_len: i64,
    pub down_sample_ratio: usize,
    pub fft_bins: Vec<f64>,
    pub x_size: usize,
    pub y_size: usize,
}

impl SdrConfig {
    pub fn default() -> Self {
        SdrConfig {
            center_freq: 103_000_000.0,
            sample_rate: 20_000_000.0,
            receive_gain: 0.0,
            timeout_us: 1_000_000,
            sample_len: 4096,
            down_sample_ratio: 2,
            x_size: 2048,
            y_size: 720,
            fft_bins: signal::fft_freqs(4096, 1.0 / 20_000_000.0),
        }
    }
}

pub fn spawn_listener(config: SdrConfig, heatmap_deque: Arc<Mutex<VecDeque<Vec<f64>>>>) {
    thread::spawn(move || {
        println!("Spawning SDR thread");
        let dev = soapysdr::Device::new("driver=hackrf").expect("HackRF not found");
        dev.set_frequency(soapysdr::Direction::Rx, 0, config.center_freq, ())
            .unwrap();
        dev.set_sample_rate(soapysdr::Direction::Rx, 0, config.sample_rate)
            .unwrap();
        dev.set_gain(soapysdr::Direction::Rx, 0, config.receive_gain)
            .unwrap();

        let mut spectral_sum: Vec<f64> = Vec::with_capacity(config.sample_len as usize);
        for _ in 0..config.sample_len {
            spectral_sum.push(0.0);
        }
        let mut spectral_density = SpectralDensity {
            spectrum: spectral_sum,
            samples: 0,
        };
        let mut rx = dev.rx_stream::<Complex<f32>>(&[0]).unwrap();
        rx.activate(None).unwrap();

        let mut planner: FftPlanner<f32> = FftPlanner::new();
        let fft = planner.plan_fft_forward(config.sample_len as usize);
        let mut buff = vec![
            Complex {
                re: 0.0f32,
                im: 0.0f32
            };
            config.sample_len as usize
        ];
        let mut data = Vec::<f64>::new();
        loop {
            spectral_density.samples += 1;
            let _ = rx.read(&mut [&mut buff], config.timeout_us).unwrap_or(0);
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
            // let avg_power = spectral_density.average_power();
            // let peaks: Vec<_> = avg_power
            //     .clone()
            //     .into_iter()
            //     .enumerate()
            //     .peaks(PeaksDetector::new(30, 5.0, 0.0), |e| e.1 as f64)
            //     .map(|((i, _), p)| (i, p))
            //     .collect();
            // let mut peaks_actual: Vec<f64> = Vec::new();
            // let mut last_idx = 0usize;
            // let mut indices: Vec<f64> = Vec::new();
            // for (index, peak_type) in peaks.iter() {
            //     if last_idx == 0 {
            //         last_idx = *index;
            //         continue;
            //     }
            //     if *index == last_idx + 1 {
            //         indices.push(avg_power[peaks[*index].0]);
            //     } else {
            //         indices.clear();
            //     }
            //     println!("{:?} {:?}", index, peak_type);
            // }
            for i in 0..config.x_size {
                let j = i * config.down_sample_ratio;
                let k = (i + 1) * config.down_sample_ratio;
                let sum: f32 = ordered_buff[j as usize..k as usize]
                    .iter()
                    .map(|x| *x)
                    .sum();
                let mean = sum / config.down_sample_ratio as f32;
                data.push(mean as f64);
            }
            // likely will need to run a notched filter over baseband
            // and capture off of center
            // data[0] = 0.0; // TODO: figure out how to handle the DC spike
            let hm = heatmap_deque.lock();
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
}
