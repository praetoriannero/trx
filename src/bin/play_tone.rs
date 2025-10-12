use rodio::{OutputStreamBuilder, Sink, Source};
use std::f32::consts::PI;
use std::time::Duration;

/// A continuous sine wave generator.
struct SineWave {
    freq: f32,
    sample_rate: u32,
    t: f32,
}

impl Iterator for SineWave {
    type Item = f32;

    fn next(&mut self) -> Option<f32> {
        let sample = (2.0 * PI * self.freq * self.t / self.sample_rate as f32).sin();
        self.t += 1.0;
        Some(sample)
    }
}

impl Source for SineWave {
    fn current_span_len(&self) -> Option<usize> {
        None
    }

    fn channels(&self) -> u16 {
        1
    }

    fn sample_rate(&self) -> u32 {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }
}

fn main() {
    let outstream = OutputStreamBuilder::open_default_stream().unwrap();
    let sink = Sink::connect_new(&outstream.mixer());

    // Create a sine wave at 440Hz
    let source = SineWave {
        freq: 400.0,
        sample_rate: 44100,
        t: 0.0,
    };

    // Add the source to the sink and play
    sink.append(source);
    sink.sleep_until_end();
}
