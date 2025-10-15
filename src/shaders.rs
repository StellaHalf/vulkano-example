use vulkano::{buffer::BufferContents, pipeline::graphics::vertex_input::Vertex};

#[derive(BufferContents, Vertex)]
#[repr(C)]
pub(crate) struct Vertex2 {
    #[format(R32G32_SFLOAT)]
    pub(crate) position: [f32; 2],
}

pub(crate) mod vs {
    use vulkano_shaders::shader;

    shader! {
        ty: "vertex",
        src: r"
            #version 460

            layout(location = 0) in vec2 position;

            void main() {
                gl_Position = vec4(position, 0.0, 1.0);
            }
        ",
    }
}

pub(crate) mod fs {
    use vulkano_shaders::shader;

    shader! {
        ty: "fragment",
        src: r"
            #version 460

            layout(location = 0) out vec4 f_color;

            void main() {
                f_color = vec4(1.0, 0.0, 0.0, 1.0);
            }
        ",
    }
}
