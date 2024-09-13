import torch
import matplotlib.pyplot as plt

def box_mask(
        data: torch.Tensor, 
        box_min=0, 
        box_max=50
    ) -> torch.Tensor:
    # Create a mask the same shape as the image
    mask = torch.zeros_like(data)
    # Fill a box area that can be edited
    mask[:, :, box_min:box_max, box_min:box_max] = 1
    return mask

def dot_mask(
        data: torch.Tensor, 
        skipped_pixels=5
    ) -> torch.Tensor:
    # Image size
    height, width = data.shape[-2], data.shape[-1]
    image = torch.zeros(1, height, width)  # Create a white image

    # Create a mask for every nth pixel horizontally and vertically
    for y in range(0, height, skipped_pixels):
        for x in range(0, width, skipped_pixels):
            image[0, y, x] = 1  # Set the pixel to white
    return image.unsqueeze(0)

def ring_mask(
        data: torch.Tensor, 
        num_rings: int = 10, 
        ring_width: int = 1, 
        ring_separation:int = 15
    ) -> torch.Tensor:
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
        image[0][mask] = 1  # Set the ring area to white
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