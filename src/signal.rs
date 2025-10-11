use num_traits::{NumCast, Zero};
use std::ops::{Add, Div};

pub fn fft_freqs(samples: u64, duration: f64) -> Vec<f64> {
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

pub struct Signal {
    pub freq_idx: usize,
    pub bandwidth: f64,
}

fn gaussian_kernel(sigma: f32) -> Vec<f32> {
    assert!(sigma > 0.0);
    let radius = (sigma * 3.0).ceil() as usize; // cover ~99.7% of area
    let size = radius * 2 + 1;
    let mut kernel = Vec::with_capacity(size);

    for i in 0..size {
        let x = i as f32 - radius as f32;
        let val = (-x * x / (2.0 * sigma * sigma)).exp();
        kernel.push(val);
    }

    // normalize
    let sum: f32 = kernel.iter().sum();
    for v in &mut kernel {
        *v /= sum;
    }
    // println!("kernel size {:?}", kernel.len());
    kernel
}

// Reflect index at the boundaries
fn reflect_index(i: isize, len: usize) -> usize {
    if len == 0 {
        return 0;
    }
    let n = len as isize;
    let period = 2 * n;
    let mut idx = ((i % period) + period) % period;
    if idx >= n {
        idx = period - 1 - idx;
    }
    idx as usize
}

/// Apply a 1D Gaussian filter to `data`, returning a new Vec<f32>.
pub fn gaussian_filter_1d(data: &Vec<f32>, sigma: f32) -> Vec<f32> {
    if sigma <= 0.0 {
        return data.clone();
    }

    let kernel = gaussian_kernel(sigma);
    let radius = kernel.len() / 2;
    let mut out = Vec::with_capacity(data.len());

    for i in 0..data.len() {
        let mut acc = 0.0;
        for (k, &w) in kernel.iter().enumerate() {
            let idx = reflect_index(i as isize + k as isize - radius as isize, data.len());
            acc += data[idx] * w;
        }
        out.push(acc);
    }

    out
}
// pub struct SpectralDensity {
//     pub spectrum: VecDeque<Vec<f64>>,
//     pub time_slices: i64,
//     pub sample_len: usize,
// }

// impl SpectralDensity {
//     pub fn average_power(&self) -> Vec<f64> {
//         let mut power: Vec<f64> = Vec::new();
//         for i in 0..spectrum.len() {
//             for j in 0..spectrum[i].len() {
//                 power
//         self.spectrum
//             .iter()
//             .map(|v| v.iter().copied().sum() / self.samples as f64)
//             .collect()
//     }
// }
//

pub fn downsample<T>(data: &Vec<T>, final_size: usize) -> (Vec<T>, Vec<usize>)
where
    T: Copy + Add<Output = T> + Div<Output = T> + NumCast + Zero,
{
    if final_size == 0 || data.is_empty() {
        return (Vec::new(), Vec::new());
    }

    let chunk_size = data.len() as f64 / final_size as f64;
    let mut result = Vec::with_capacity(final_size);
    let mut map = Vec::with_capacity(data.len());

    for i in 0..final_size {
        let start = (i as f64 * chunk_size).floor() as usize;
        let end = ((i as f64 + 1.0) * chunk_size).floor() as usize;
        let end = end.min(data.len());

        if start < end {
            let mut sum = T::zero();
            for &x in &data[start..end] {
                sum = sum + x;
            }

            for _ in start..end {
                map.push(i);
            }

            let avg = sum / NumCast::from(end - start).unwrap();
            result.push(avg);
        } else {
            result.push(*data.last().unwrap());
        }
    }

    (result, map)
}
