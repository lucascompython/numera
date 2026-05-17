use gpui::*;
use mimalloc::MiMalloc;

#[global_allocator]
static GLOBAL: MiMalloc = MiMalloc;

mod app;
mod autonomous_numbering;
mod numbering_mode;
mod numbering_session;
mod ocr;
mod processing;

fn main() {
    // Initialize OCR engine in background (may take a moment to load models)
    std::thread::spawn(|| {
        if let Err(e) = ocr::init_ocr() {
            eprintln!("OCR initialization failed: {e}");
        } else {
            eprintln!("OCR engine initialized successfully");
        }
    });

    let app = gpui_platform::application().with_assets(gpui_component_assets::Assets);

    app.run(move |cx| {
        gpui_component::init(cx);

        let opts = gpui::WindowOptions {
            titlebar: Some(gpui::TitlebarOptions {
                title: Some("Numera".into()),
                ..Default::default()
            }),
            ..Default::default()
        };

        cx.open_window(opts, |window, cx| {
            let view = cx.new(|cx| app::App::new(window, cx));

            gpui_component::Theme::change(gpui_component::ThemeMode::Dark, Some(window), cx);
            cx.new(|cx| gpui_component::Root::new(view, window, cx))
        })
        .unwrap();
    });
}
