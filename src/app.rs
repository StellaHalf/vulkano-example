use std::sync::Arc;

use vulkano::instance::Instance;
use winit::{
    application::ApplicationHandler, event::WindowEvent, event_loop::ActiveEventLoop,
    window::Window,
};

use crate::renderer::Renderer;

pub(crate) struct App {
    instance: Arc<Instance>,
    renderer: Option<Renderer>,
}

impl App {
    pub(crate) fn new(instance: &Arc<Instance>) -> Self {
        App {
            instance: instance.clone(),
            renderer: None,
        }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        if self.renderer.is_none() {
            let window = event_loop
                .create_window(Window::default_attributes())
                .unwrap();
            self.renderer = Some(Renderer::init(&self.instance, &Arc::new(window)));
        }
    }

    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        _window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::RedrawRequested => self.renderer.as_mut().unwrap().draw(),
            _ => (),
        }
    }
}
