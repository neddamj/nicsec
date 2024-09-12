import torch
import matplotlib.pyplot as plt

def box_mask(data, box_min=0, box_max=50):
    # Create a mask the same shape as the image
    mask = torch.zeros_like(data)
    # Fill a box area that can be edited
    mask[:, :, box_min:box_max, box_min:box_max] = 1
    return mask

def ring_mask(data, num_rings=10, ring_width=1, ring_separation=15):
    # Image size
    height, width = data.shape[-2], data.shape[-1]
    image = torch.zeros(1, height, width)  # Create a black image (1 channel)

    # Coordinates grid
    y = torch.arange(height).unsqueeze(1).repeat(1, width)
    x = torch.arange(width).unsqueeze(0).repeat(height, 1)

    # Center of the image
    center_y, center_x = height // 2, width // 2

    # Calculate the distance from the center for each pixel
    distance_from_center = torch.sqrt((x - center_x)**2 + (y - center_y)**2)

    for i in range(num_rings):
        inner_radius = ring_separation * i * ring_width
        outer_radius = inner_radius + ring_width
        mask = (distance_from_center >= inner_radius) & (distance_from_center < outer_radius)
        image[0][mask] = 1  # Set the ring area to black (0)
    return image.unsqueeze(0)

if __name__ == "__main__":
    # Create random noise in the shape of an image
    img = torch.rand(1, 3, 128, 128)
    # Generate mask
    image = ring_mask()
    # Convert tensor to numpy for visualization
    image_np = image.squeeze().numpy()

    # Display the image
    plt.imshow(image_np, cmap='gray')
    plt.axis('off')
    plt.show()