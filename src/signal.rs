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
