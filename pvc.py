# Import necessary libraries
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# -----------------------------
# 1. Load and Preprocess the MNIST Dataset
# -----------------------------

# Define a set of image transformations:
# - Convert PIL images to PyTorch tensors
# - Resize images from 28x28 to 16x16 (simulating a lower resolution)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((16, 16))
])

# Download the MNIST training dataset and apply the transform
train_data = datasets.MNIST(root='./data', train=True, download=True, transform=transform)

# Create a data loader to iterate over the dataset in batches
loader = torch.utils.data.DataLoader(train_data, batch_size=32, shuffle=True)

# -----------------------------
# 2. Define the Autoencoder Model
# -----------------------------

# An autoencoder compresses input data (encoder) and then reconstructs it (decoder)
class AutoEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        
        # Encoder: reduces image from 16x16 to 4x4 using 2 convolutional layers
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 8, 3, stride=2, padding=1),  # Output: (8 channels, 8x8)
            nn.ReLU(),
            nn.Conv2d(8, 16, 3, stride=2, padding=1), # Output: (16 channels, 4x4)
            nn.ReLU()
        )
        
        # Decoder: reconstructs image back to 16x16 using transposed convolution
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(16, 8, 4, stride=2, padding=1),  # Output: (8, 8x8)
            nn.ReLU(),
            nn.ConvTranspose2d(8, 1, 4, stride=2, padding=1),   # Output: (1, 16x16)
            nn.Sigmoid()  # Keep pixel values between 0 and 1
        )

    # Forward pass: encode input and then decode it
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded

# -----------------------------
# 3. Train the Autoencoder
# -----------------------------

# Instantiate the model
model = AutoEncoder()

# Define the optimizer (Adam) and loss function (MSE for image reconstruction)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()

# Training loop for a few epochs (can increase for better results)
print("Training Autoencoder...")
for epoch in range(2):
    for imgs, _ in loader:  # We ignore labels (_) since this is unsupervised
        recon = model(imgs)               # Forward pass: reconstruct images
        loss = criterion(recon, imgs)     # Compute reconstruction loss
        optimizer.zero_grad()             # Clear gradients
        loss.backward()                   # Backpropagate the error
        optimizer.step()                  # Update model weights
    print(f"Epoch {epoch+1} Loss: {loss.item():.4f}")

# -----------------------------
# 4. Visualize Original vs Reconstructed Images
# -----------------------------

# Get a batch of test images from the loader
test_imgs, _ = next(iter(loader))

# Use the trained model to reconstruct these images
reconstructed = model(test_imgs)

# Plot original and reconstructed images side by side
fig, axs = plt.subplots(2, 6, figsize=(10, 4))
for i in range(6):
    # Display original image (top row)
    axs[0, i].imshow(test_imgs[i][0].detach().numpy(), cmap='gray')
    axs[0, i].axis('off')
    
    # Display reconstructed image (bottom row)
    axs[1, i].imshow(reconstructed[i][0].detach().numpy(), cmap='gray')
    axs[1, i].axis('off')

# Add titles
axs[0, 0].set_title('Original')
axs[1, 0].set_title('Reconstructed')

# Save the figure to a file
plt.tight_layout()
plt.savefig("reconstruction_result.png")
print("Saved image to reconstruction_result.png ✅")

