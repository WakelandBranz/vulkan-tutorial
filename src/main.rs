pub mod shaders;

use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    descriptor_set::{
        DescriptorSet, WriteDescriptorSet, allocator::StandardDescriptorSetAllocator,
    },
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    pipeline::{
        ComputePipeline, Pipeline, PipelineLayout, PipelineShaderStageCreateInfo,
        compute::ComputePipelineCreateInfo, layout::PipelineDescriptorSetLayoutCreateInfo,
    },
    sync::{self, GpuFuture},
};

use crate::shaders::cs;

fn main() {
    vulkano_tutorial().expect("Failed to initialize vulkano using 'init_vulkano()'");
}

// Used Claude for code commenting so that I can come back to this and understand
fn vulkano_tutorial() -> Result<(), Box<dyn std::error::Error>> {
    // The InstanceCreateFlags::ENUMERATE_PORTABILITY flag is set to support devices, such as those on MacOS and iOS systems, that do not fully conform to the Vulkan Specification
    // Load the Vulkan library
    let library = VulkanLibrary::new()?;
    // Create Vulkan instance - entry point for all Vulkan operations
    let instance = Instance::new(
        library,
        InstanceCreateInfo {
            flags: InstanceCreateFlags::ENUMERATE_PORTABILITY,
            ..Default::default()
        },
    )?;

    // Find a physical device which we can use to render (iGPU, GeForce/Radeon graphics cards, etc.)
    let physical_device = instance
        .enumerate_physical_devices()
        .expect("Could not enumerate physical devices!")
        .next()
        .expect("No physical devices available!");

    // Gather the index of a viable queue family
    // Queue families group queues with similar capabilities (graphics, compute, transfer)
    let queue_family_index = physical_device
        .queue_family_properties()
        .iter()
        .enumerate()
        .position(|(_queue_family_index, queue_family_properties)| {
            queue_family_properties
                .queue_flags
                .contains(QueueFlags::GRAPHICS)
        })
        .expect("Couldn't find a graphical queue family") as u32;

    // Print device information
    println!(
        "Successfully chosen device {:?} running driver {:?} with version {:?}",
        physical_device.properties().device_name,
        physical_device.properties().driver_name.as_ref().unwrap(),
        physical_device.properties().driver_version
    );

    // Create logical device and queues from physical device
    let (device, mut queues) = Device::new(
        physical_device,
        DeviceCreateInfo {
            // Here we pass the desired queue family to use by index
            queue_create_infos: vec![QueueCreateInfo {
                queue_family_index,
                ..Default::default()
            }],
            ..Default::default()
        },
    )
    .expect("Failed to create a device!");

    // We now have an open channel of communication with a Vulkan device!
    // That being said, 'queues' is an iterator, but in this case it is just one device so we must extract it.
    let queue = queues.next().unwrap();

    // Remember, cloning device just clones the Arc which is inexpensive.
    // Create memory allocator for GPU memory management
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/allocator/trait.CommandBufferAllocator.html
    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/allocator/struct.StandardCommandBufferAllocator.html
    // TODO!: read more about secondary command buffers which can be found below
    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/index.html
    // Create command buffer allocator - manages command buffer memory pools
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    // This is how to use command buffers. Use it when you render a frame.
    // Use CommandBufferUsage::OneTimeSubmit for dynamic frames, use CommandBufferUsage::MultipleSubmit for static things like UIs.
    // Create a primary command buffer builder
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )?;

    // Build the command buffer (currently empty)
    let command_buffer = Arc::new(command_buffer_builder.build()?);

    // Create source buffer with values 0..64
    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC, // Can be used as copy source
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST // CPU-accessible memory
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE, // Optimized for sequential writes
            ..Default::default()
        },
        source_content,
    )
    .expect("failed to create source buffer");

    // Create destination buffer filled with zeros
    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST, // Can be used as copy destination
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST // CPU-accessible memory
                | MemoryTypeFilter::HOST_RANDOM_ACCESS, // Optimized for random access reads
            ..Default::default()
        },
        destination_content,
    )
    .expect("failed to create destination buffer");

    // Create another command buffer allocator (reusing same one would be better practice)
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    /*
    We create a builder, add a copy command to it with copy_buffer, then turn that builder into an actual command buffer with .build().
    Like we saw in the buffers creation section, we call .clone() multiple times, but we only clone Arcs.
    */
    // Create command buffer builder for copy operation
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )?;

    // Record copy command - copy from source to destination buffer
    builder.copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))?;

    // Build the command buffer with recorded copy command
    let command_buffer = builder.build().unwrap();

    /*
     */
    //sync::now(device.clone())
    //    .then_execute(queue.clone(), command_buffer.clone())
    //    .unwrap()
    //    .flush()?;

    // Submit command buffer to GPU queue with fence for synchronization
    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()?; // same as signal fence, and then flush

    // Wait for GPU to complete the copy operation
    future.wait(None)?;

    // Read source buffer contents from GPU memory to CPU
    let src_content = source.read().unwrap();
    // Read destination buffer contents from GPU memory to CPU
    let destination_content = destination.read().unwrap();
    // Verify the copy operation worked - both buffers should have identical data
    assert_eq!(&*src_content, &*destination_content);

    // End of section 3 of vulkan.rs book

    // Create buffer with 65536 u32 values (0..65536) in GPU memory
    // STORAGE_BUFFER = can be read/written by compute shaders
    // PREFER_DEVICE = GPU memory (fast for GPU, slower for CPU access)
    let data_iter = 0..65536u32;
    let data_buffer = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::STORAGE_BUFFER,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_DEVICE
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        data_iter,
    )?;

    // Loading the shader in shader.rs
    // Load compiled SPIR-V compute shader module
    let shader = cs::load(device.clone())?;

    // Get the "main" entry point function from the shader
    let cs = shader.entry_point("main").unwrap();

    // Create shader stage (tells Vulkan this is a compute shader)
    let stage = PipelineShaderStageCreateInfo::new(cs);

    // Create pipeline layout - defines what resources (buffers, uniforms) shader expects
    // Automatically introspects shader to determine descriptor set layout
    let layout = PipelineLayout::new(
        device.clone(),
        PipelineDescriptorSetLayoutCreateInfo::from_stages([&stage])
            .into_pipeline_layout_create_info(device.clone())
            .unwrap(),
    )?;

    // Create the compute pipeline - combines shader + layout into executable GPU program
    let compute_pipeline = ComputePipeline::new(
        device.clone(),
        None, // No pipeline cache
        ComputePipelineCreateInfo::stage_layout(stage, layout),
    )?;

    let descriptor_set_allocator = Arc::new(StandardDescriptorSetAllocator::new(
        device.clone(),
        Default::default(),
    ));
    let pipeline_layout = compute_pipeline.layout();
    let descriptor_set_layouts = pipeline_layout.set_layouts();

    let descriptor_set_layout_index = 0;
    let descriptor_set_layout = descriptor_set_layouts
        .get(descriptor_set_layout_index)
        .unwrap();
    let descriptor_set = DescriptorSet::new(
        descriptor_set_allocator,
        descriptor_set_layout.clone(),
        [WriteDescriptorSet::buffer(0, data_buffer.clone())], // 0 is the binding
        [],
    )?;

    // Start on section 4.4 next. Too tired tonight.

    println!("VulkanRenderer setup successful.");

    Ok(())
}
