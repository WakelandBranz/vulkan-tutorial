use std::sync::Arc;

use vulkano::{
    VulkanLibrary,
    buffer::{Buffer, BufferCreateInfo, BufferUsage},
    command_buffer::{
        AutoCommandBufferBuilder, CommandBufferUsage, CopyBufferInfo,
        allocator::{StandardCommandBufferAllocator, StandardCommandBufferAllocatorCreateInfo},
    },
    device::{Device, DeviceCreateInfo, Queue, QueueCreateInfo, QueueFlags},
    instance::{Instance, InstanceCreateFlags, InstanceCreateInfo},
    memory::allocator::{AllocationCreateInfo, MemoryTypeFilter, StandardMemoryAllocator},
    sync::{self, GpuFuture},
};

fn main() {
    init_vulkano().expect("Failed to initialize vulkano using 'init_vulkano()'");
}

fn init_vulkano() -> Result<(), Box<dyn std::error::Error>> {
    // The InstanceCreateFlags::ENUMERATE_PORTABILITY flag is set to support devices, such as those on MacOS and iOS systems, that do not fully conform to the Vulkan Specification
    let library = VulkanLibrary::new()?;
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

    println!(
        "Successfully chosen device {:?} running driver {:?} with version {:?}",
        physical_device.properties().device_name,
        physical_device.properties().driver_name.as_ref().unwrap(),
        physical_device.properties().driver_version
    );

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
    let memory_allocator = Arc::new(StandardMemoryAllocator::new_default(device.clone()));

    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/allocator/trait.CommandBufferAllocator.html
    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/allocator/struct.StandardCommandBufferAllocator.html
    // TODO!: read more about secondary command buffers which can be found below
    // https://docs.rs/vulkano/0.34.0/vulkano/command_buffer/index.html
    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    // This is how to use command buffers. Use it when you render a frame.
    // Use CommandBufferUsage::OneTimeSubmit for dynamic frames, use CommandBufferUsage::MultipleSubmit for static things like UIs.
    let mut command_buffer_builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator.clone(),
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )?;

    let command_buffer = Arc::new(command_buffer_builder.build()?);

    let source_content: Vec<i32> = (0..64).collect();
    let source = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_SRC,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_SEQUENTIAL_WRITE,
            ..Default::default()
        },
        source_content,
    )
    .expect("failed to create source buffer");

    let destination_content: Vec<i32> = (0..64).map(|_| 0).collect();
    let destination = Buffer::from_iter(
        memory_allocator.clone(),
        BufferCreateInfo {
            usage: BufferUsage::TRANSFER_DST,
            ..Default::default()
        },
        AllocationCreateInfo {
            memory_type_filter: MemoryTypeFilter::PREFER_HOST
                | MemoryTypeFilter::HOST_RANDOM_ACCESS,
            ..Default::default()
        },
        destination_content,
    )
    .expect("failed to create destination buffer");

    let command_buffer_allocator = Arc::new(StandardCommandBufferAllocator::new(
        device.clone(),
        StandardCommandBufferAllocatorCreateInfo::default(),
    ));

    /*
    We create a builder, add a copy command to it with copy_buffer, then turn that builder into an actual command buffer with .build().
    Like we saw in the buffers creation section, we call .clone() multiple times, but we only clone Arcs.
    */
    let mut builder = AutoCommandBufferBuilder::primary(
        command_buffer_allocator,
        queue_family_index,
        CommandBufferUsage::OneTimeSubmit,
    )?;

    builder.copy_buffer(CopyBufferInfo::buffers(source.clone(), destination.clone()))?;

    let command_buffer = builder.build().unwrap();

    /*
     */
    //sync::now(device.clone())
    //    .then_execute(queue.clone(), command_buffer.clone())
    //    .unwrap()
    //    .flush()?;

    let future = sync::now(device.clone())
        .then_execute(queue.clone(), command_buffer)
        .unwrap()
        .then_signal_fence_and_flush()?; // same as signal fence, and then flush

    future.wait(None)?;

    let src_content = source.read().unwrap();
    let destination_content = destination.read().unwrap();
    assert_eq!(&*src_content, &*destination_content);

    // End of section 3 of vulkan.rs book

    println!("VulkanRenderer setup successful.");

    Ok(())
}
