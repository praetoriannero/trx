use num_complex::ComplexFloat;
use rustfft::{FftPlanner, num_complex::Complex};
use smoothed_z_score::{Peak, PeaksDetector, PeaksFilter};
use statrs::statistics::Statistics as Stats;
use std::collections::VecDeque;
use std::sync::{Arc, Mutex, mpsc};
use std::{thread, time};

use crate::signal::{Signal, downsample, fft_freqs, gaussian_filter_1d};

#[derive(Clone)]
pub struct SdrConfig {
    pub center_freq: f64,
    pub sample_rate: f64,
    pub receive_gain: f64,
    pub timeout_us: i64,
    pub sample_len: i64,
    pub fft_bins: Vec<f64>,
    pub x_size: usize,
    pub y_size: usize,
}

impl SdrConfig {
    pub fn default() -> Self {
        SdrConfig {
            center_freq: 103_000_000.0,
            sample_rate: 10_000_000.0,
            receive_gain: 0.0,
            timeout_us: 1_000_000,
            sample_len: 4096,
            x_size: 1024,
            y_size: 720,
            fft_bins: fft_freqs(4096, 1.0 / 10_000_000.0),
        }
    }
}

fn find_peaks(spectrum: Arc<Mutex<Vec<f32>>>, found_signals: Arc<Mutex<Vec<Signal>>>) {
    thread::spawn(move || {
        loop {
            let delay = 1;
            thread::sleep(time::Duration::from_secs(delay));
            let mut avg_power = vec![0.0_f32; 4096];
            // let slices = 20;
            // for row in 0..slices {
            //     for col in 0..spectrum[row].len() {
            //         avg_power[col] += spectrum[row][col];
            //     }
            // }
            // for col in 0..avg_power.len() {
            //     avg_power[col] /= slices as f32;
            // }
            //
            // let peaks: Vec<_> = avg_power
            //     .clone()
            //     .into_iter()
            //     .enumerate()
            //     .peaks(PeaksDetector::new(30, 5.0, 0.0), |e| e.1 as f64)
            //     .map(|((i, _), p)| (i, p))
            //     .collect();
        }
    });
}

pub fn spawn_listener(
    config: SdrConfig,
    heatmap_deque: Arc<Mutex<VecDeque<Vec<f64>>>>,
    found_signals: Arc<Mutex<Vec<Signal>>>,
) {
    thread::spawn(move || {
        println!("Spawning SDR thread");
        let dev = soapysdr::Device::new("driver=hackrf").expect("HackRF not found");
        dev.set_frequency(soapysdr::Direction::Rx, 0, config.center_freq, ())
            .unwrap();
        dev.set_sample_rate(soapysdr::Direction::Rx, 0, config.sample_rate)
            .unwrap();
        dev.set_gain(soapysdr::Direction::Rx, 0, config.receive_gain)
            .unwrap();

        let mut spectrum: VecDeque<Vec<f32>> = VecDeque::new();
        for _ in 0..config.y_size {
            let mut row: Vec<f32> = Vec::new();
            for _ in 0..config.sample_len {
                row.push(0.0);
            }
            spectrum.push_front(row);
        }
        let mut spectral_sum: Vec<f64> = Vec::with_capacity(config.sample_len as usize);
        for _ in 0..config.sample_len {
            spectral_sum.push(0.0);
        }
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
        let mut heartbeat = time::SystemTime::now();

        loop {
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
            spectrum.pop_front();
            spectrum.push_back(ordered_buff.iter().copied().collect());
            let (data, idx_map) = downsample::<f32>(&ordered_buff, config.x_size);

            let mut avg_power = vec![0.0_f32; 4096];
            let slices = config.y_size;
            // if total_iters % 1_000 == 0 {
            if heartbeat.elapsed().unwrap() > time::Duration::new(1, 0) {
                for row in 0..slices {
                    for col in 0..spectrum[row].len() {
                        avg_power[col] += spectrum[row][col];
                    }
                }
                for col in 0..avg_power.len() {
                    avg_power[col] /= slices as f32;
                }
                avg_power = gaussian_filter_1d(&avg_power, 10.0);
                println!(
                    "min power {}",
                    avg_power
                        .iter()
                        .fold(f32::INFINITY, |current_min, &val| current_min.min(val))
                );
                println!(
                    "max power {}",
                    avg_power
                        .iter()
                        .fold(f32::NEG_INFINITY, |current_max, &val| current_max.max(val))
                );
                let mut peaks = Vec::<(usize, Peak)>::new();
                let threshold = 0.3;
                for idx in 0..avg_power.len() {
                    if avg_power[idx] > threshold {
                        peaks.push((idx, Peak::High));
                    }
                }
                // let peaks: Vec<_> = avg_power
                //     .clone()
                //     .into_iter()
                //     .enumerate()
                //     .peaks(PeaksDetector::new(256, 8.0, 0.0), |e| e.1 as f64)
                //     .map(|((i, _), p)| (i, p))
                //     .collect();
                let mut peaks_actual: Vec<usize> = Vec::new();
                let mut indices: Vec<usize> = Vec::new();
                for (index, peak_type) in peaks.iter() {
                    if *peak_type == Peak::Low {
                        continue;
                    }
                    if indices.len() > 0 {
                        if *indices.last().unwrap() == *index - 1 {
                            indices.push(*index);
                            continue;
                        } else {
                            println!("{:?}", indices);
                            let index_precise: f32 =
                                indices.iter().sum::<usize>() as f32 / indices.len() as f32;
                            let index_ceil = index_precise as usize;
                            peaks_actual.push(index_ceil);
                            indices.clear();
                        }
                    }
                    indices.push(*index);
                }

                println!("PA {:?}", peaks_actual);
                {
                    let mut signals = found_signals.lock().unwrap();
                    signals.clear();
                    for index in peaks_actual {
                        signals.push(Signal {
                            freq_idx: idx_map[index],
                            center_frequency: config.fft_bins[index],
                            bandwidth: 0.0,
                        });
                    }
                }
                heartbeat = time::SystemTime::now();
            }

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
