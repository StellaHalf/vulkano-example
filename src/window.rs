use vulkano::{
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    swapchain::Surface,
};
use winit::event_loop::{self, EventLoop};

use crate::app::App;

pub(crate) fn launch() {
    let library = vulkano::VulkanLibrary::new().expect("no lokal Vulkan library/DLL");

    let event_loop = EventLoop::new().unwrap();
    event_loop.set_control_flow(event_loop::ControlFlow::Poll);

    let required_extensions = Surface::required_extensions(&event_loop).unwrap();
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            enabled_extensions: required_extensions,
            ..Default::default()
        },
    )
    .expect("failed to create instance");

    let mut app = App::new(&instance);
    event_loop.run_app(&mut app).unwrap();
}
