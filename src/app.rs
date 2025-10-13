use egui::{Align2, Color32, ColorImage, FontFamily, FontId, Image, Rect, Stroke};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

use crate::signal::Signal;

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

pub struct HeatmapApp {
    pub buffer: Arc<Mutex<VecDeque<Vec<f64>>>>,
    pub time_series: Arc<Mutex<Vec<f64>>>,
    pub x_size: i64,
    pub y_size: i64,
    pub threshold: f64,
    pub detected_signals: Arc<Mutex<Vec<Signal>>>,
    pub center_frequency: f32,
}

trait Render {
    fn draw_spectrogram(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response>;
    fn draw_constellation(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response>;
    fn draw_time_series(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response>;
    fn draw_controls(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response>;
}

impl Render for HeatmapApp {
    fn draw_spectrogram(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response> {
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
                        return None;
                    }
                    let pixel = row.unwrap().get(x as usize);
                    if pixel.is_none() {
                        return None;
                    }
                    let v = pixel.unwrap();
                    let intensity = (sigmoid(*v) * 255.0) as u8;
                    img[(x as usize, y as usize)] =
                        egui::Color32::from_rgb(intensity, intensity, 0);
                }
            }
        }
        let text_color = Color32::from_rgb(255, 0, 0);
        let tex = ctx.load_texture("spectrogram", img, Default::default());
        let sized_tex = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
        let spec_tex = Image::new(sized_tex);
        let stroke = Stroke::new(1.0, Color32::from_rgb(0, 127, 255));
        let bw_stroke = Stroke::new(1.0, Color32::from_rgb(0, 127, 0));
        // Start draw
        let win_response = egui::Window::new("Spectrogram")
            .fixed_pos(parent_rect.left_bottom())
            .collapsible(false)
            .show(ctx, |ui| {
                let spec_response = ui.add(spec_tex);
                let spec_rect: Rect = spec_response.rect;
                let x_offset = spec_rect.left();
                let y_min = spec_rect.top();
                let y_max = spec_rect.bottom();
                let y_range = y_min..=y_max;
                let signals = self.detected_signals.lock().unwrap();

                for signal in signals.iter() {
                    let bin = signal.freq_idx;
                    ui.painter()
                        .vline(bin as f32 + x_offset, y_range.clone(), stroke);
                    ui.painter().vline(
                        signal.lower_freq_idx as f32 + x_offset,
                        y_range.clone(),
                        bw_stroke,
                    );
                    ui.painter().vline(
                        signal.upper_freq_idx as f32 + x_offset,
                        y_range.clone(),
                        bw_stroke,
                    );
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
        Some(win_response.unwrap().response)
    }

    fn draw_time_series(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response> {
        let mut response: Option<egui::Response> = None;
        let bw_stroke = Stroke::new(1.0, Color32::from_rgb(0, 127, 0));
        egui::Window::new("FFT")
            .fixed_pos(parent_rect.left_bottom())
            .collapsible(false)
            .show(ctx, |ui| {
                let x_size = self.x_size as usize;
                let y_size = self.y_size as usize;
                let img = ColorImage::new(
                    [x_size, y_size],
                    vec![egui::Color32::BLACK; x_size * y_size as usize],
                );

                let tex = ctx.load_texture("time-series", img, Default::default());
                let sized_tex = egui::load::SizedTexture::new(tex.id(), tex.size_vec2());
                let spec_tex = Image::new(sized_tex);
                response = Some(ui.add(spec_tex));
                let spec_rect = response.clone().unwrap().rect;
                let x_offset = spec_rect.left();
                let y_offset = spec_rect.bottom();
                {
                    let sig_vec = self.time_series.lock().unwrap();
                    let thickness = 2 as f32;
                    let color = Color32::from_rgb(255, 255, 0);
                    let points: Vec<egui::Pos2> = (0..sig_vec.len())
                        .map(|idx| {
                            egui::Pos2::new(idx as f32 + x_offset, -sig_vec[idx] as f32 + y_offset)
                        })
                        .collect();
                    let shape = egui::epaint::Shape::line(
                        points,
                        egui::epaint::PathStroke::new(thickness, color),
                    );
                    ui.painter().add(shape);
                }
                ui.painter().hline(
                    spec_rect.left()..=spec_rect.right(),
                    spec_rect.bottom() - (self.threshold as f32 * 10.0),
                    bw_stroke,
                );
            });
        response
    }

    fn draw_constellation(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response> {
        None
    }

    fn draw_controls(
        &mut self,
        ctx: &egui::Context,
        _: &mut eframe::Frame,
        parent_rect: &egui::Rect,
    ) -> Option<egui::Response> {
        None
    }
}

impl eframe::App for HeatmapApp {
    fn update(&mut self, ctx: &egui::Context, frame: &mut eframe::Frame) {
        let mut parent_rect = egui::Rect::from_pos(egui::Pos2::new(0.0, 0.0));
        let spec_response = self.draw_spectrogram(ctx, frame, &parent_rect);
        match spec_response {
            Some(resp) => parent_rect = resp.rect,
            None => (),
        }
        self.draw_time_series(ctx, frame, &parent_rect);
        self.draw_constellation(ctx, frame, &parent_rect);
        self.draw_controls(ctx, frame, &parent_rect);
        let mut inner_size: egui::Vec2 = ctx.used_size();
        inner_size.y += 5.0;
        ctx.send_viewport_cmd(egui::ViewportCommand::InnerSize(inner_size));
        ctx.request_repaint();
    }
}
