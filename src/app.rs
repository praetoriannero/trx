use egui::{Align2, CentralPanel, Color32, ColorImage, FontFamily, FontId, Image, Rect, Stroke};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::signal::Signal;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct HeatmapApp {
    pub buffer: Arc<Mutex<VecDeque<Vec<f64>>>>,
    pub x_size: i64,
    pub y_size: i64,
    pub detected_signals: Arc<Mutex<Vec<Signal>>>,
    pub center_frequency: f32,
}

impl eframe::App for HeatmapApp {
    fn update(&mut self, ctx: &egui::Context, _: &mut eframe::Frame) {
        let mut img = ColorImage::new(
            [self.x_size as usize, self.y_size as usize],
            vec![egui::Color32::BLACK; self.x_size as usize * self.y_size as usize],
        );
        {
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
                    img[(x as usize, y as usize)] =
                        egui::Color32::from_rgb(intensity, intensity, 0);
                }
            }
        }
        // let font_family = FontFamily::Monospace;
        // let font_id = FontId::new(11.0, font_family);
        let text_color = Color32::from_rgb(255, 0, 0);
        let tex = ctx.load_texture("spectrogram", img, Default::default());
        let sized_tex = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
        let spec_tex = Image::new(sized_tex);
        let stroke = Stroke::new(1.0, Color32::from_rgb(0, 127, 255));

        // Start draw
        CentralPanel::default().show(ctx, |ui| {
            let response = ui.add(spec_tex);
            let spec_rect: Rect = response.rect;
            // let mut points = vec![spec_rect.left_top(), spec_rect.right_bottom()];
            // ui.painter().line(points, stroke);
            // points = vec![spec_rect.left_bottom(), spec_rect.right_top()];
            // ui.painter().line(points, stroke);
            // ui.painter().text(
            //     spec_rect.center(),
            //     Align2::CENTER_CENTER,
            //     "test",
            //     font_id,
            //     text_color,
            // );
            let x_offset = spec_rect.left();
            let y_min = spec_rect.top();
            let y_max = spec_rect.bottom();
            let signals = self.detected_signals.lock().unwrap();
            for signal in signals.iter() {
                let bin = signal.freq_idx;
                ui.painter()
                    .vline(bin as f32 + x_offset, y_min..=y_max, stroke);
                let text_pos = egui::Pos2 {
                    x: bin as f32 + x_offset,
                    y: spec_rect.center().y,
                };
                let font_family = FontFamily::Monospace;
                let font_id = FontId::new(12.0, font_family);
                ui.painter().text(
                    text_pos,
                    Align2::CENTER_CENTER,
                    format!(
                        "{:3.3}",
                        (self.center_frequency as f64 + signal.center_frequency) / 1_000_000.0
                    ),
                    font_id,
                    text_color,
                );
            }
        });

        ctx.request_repaint();
    }
}
