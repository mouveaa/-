# Diffusion model implementation for CIFAR10 
# Based on DDPM paper but simplified for learning/testing
# Note: Might need to tune parameters for better results

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from torchmetrics.image.fid import FrechetInceptionDistance

use_cuda = torch.cuda.is_available()
device = torch.device("cuda" if use_cuda else "cpu")
print(f"Using device: {device}")  # debug info

# Hyperparams - might need tweaking
BATCH_SIZE = 128    # decent size for my GPU memory
LEARNING_RATE = 1e-4  # standard Adam lr
EPOCHS = 5          # starting small, can increase if results look promising
IMG_SIZE = 32       # CIFAR10 size
TIMESTEPS = 50      # fewer steps for faster training, might need more for quality

# Mixed precision training - helps with memory/speed on newer GPUs
scaler = torch.cuda.amp.GradScaler() if use_cuda else None

# Basic transforms - keeping it simple for now
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))  # normalize to [-1,1]
])

# Get CIFAR10 - download if needed
train_dataset = torchvision.datasets.CIFAR10(
    root='./data', 
    train=True, 
    transform=transform, 
    download=True
)

if __name__ == '__main__':
    # DataLoader with some performance optimizations
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=BATCH_SIZE, 
        shuffle=True,
        num_workers=4,  # adjust based on your CPU
        pin_memory=True  # helps with GPU transfer
    )

    # Simple U-Net - probably need to make this more sophisticated later
    class UNet(nn.Module):
        def __init__(self):
            super().__init__()
            
            # Encoder - going deeper might help but needs more memory
            self.encoder = nn.Sequential(
                nn.Conv2d(3, 64, 3, 1, 1), nn.ReLU(),
                nn.Conv2d(64, 128, 3, 2, 1), nn.ReLU(),  # /2
                nn.Conv2d(128, 256, 3, 2, 1), nn.ReLU()  # /4
            )
            
            # Simple bottleneck - might add attention here later
            self.bottleneck = nn.Sequential(
                nn.Conv2d(256, 256, 3, 1, 1), 
                nn.ReLU()
            )
            
            # Decoder - matching encoder structure
            self.decoder = nn.Sequential(
                nn.ConvTranspose2d(256, 128, 4, 2, 1), nn.ReLU(),  # *2
                nn.ConvTranspose2d(128, 64, 4, 2, 1), nn.ReLU(),   # *4
                nn.Conv2d(64, 3, 3, 1, 1)  # back to RGB
            )

        def forward(self, x):
            # TODO: Add skip connections?
            x = self.encoder(x)
            x = self.bottleneck(x)
            return self.decoder(x)

    model = UNet().to(device)
    criterion = nn.MSELoss()  # simple MSE for now
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Noise schedule - linear for simplicity
    # Note: Could try cosine schedule like improved DDPM paper
    betas = torch.linspace(1e-4, 0.02, TIMESTEPS).to(device)
    alphas = 1.0 - betas
    alpha_hat = torch.cumprod(alphas, dim=0)

    def forward_diffusion(x0, t):
        # Add noise according to diffusion schedule
        noise = torch.randn_like(x0).to(device)
        alpha_t = alpha_hat[t].view(-1, 1, 1, 1)
        return (
            torch.sqrt(alpha_t) * x0 + torch.sqrt(1 - alpha_t) * noise,
            noise
        )

    # Training loop with loss tracking
    train_losses = []
    best_loss = float('inf')  # track best model

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        batch_count = 0
        
        for batch_idx, (images, _) in enumerate(train_loader):
            images = images.to(device)
            # Random timesteps for each image
            t = torch.randint(0, TIMESTEPS, (images.shape[0],), device=device)
            
            # Get noisy images and target noise
            noisy_images, noise = forward_diffusion(images, t)
            
            optimizer.zero_grad()
            
            # Use AMP if available
            if use_cuda:
                with torch.cuda.amp.autocast():
                    pred_noise = model(noisy_images)
                    loss = criterion(pred_noise, noise)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                pred_noise = model(noisy_images)
                loss = criterion(pred_noise, noise)
                loss.backward()
                optimizer.step()

            epoch_loss += loss.item()
            batch_count += 1
            
            # Print progress
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")

        avg_loss = epoch_loss / batch_count
        train_losses.append(avg_loss)
        
        print(f"Epoch [{epoch+1}/{EPOCHS}], Avg Loss: {avg_loss:.4f}")
        
        # Save best model - might want to use this later
        if avg_loss < best_loss:
            best_loss = avg_loss
            # torch.save(model.state_dict(), 'best_diffusion.pth')

    # Visualize training progress
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, 'b-', label="Training Loss")
    plt.xlabel("Epochs")
    plt.ylabel("MSE Loss")
    plt.title("Diffusion Model Training Progress")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Sampling function - this is where the magic happens
    @torch.no_grad()
    def sample_images(num_samples=16):
        model.eval()
        # Start from random noise
        x = torch.randn((num_samples, 3, IMG_SIZE, IMG_SIZE)).to(device)
        
        # Gradually denoise
        for t in range(TIMESTEPS - 1, -1, -1):
            # Add noise at each step (except last)
            z = torch.randn_like(x).to(device) if t > 0 else 0
            
            # One step of denoising
            x = (1 / torch.sqrt(alphas[t])) * (
                x - ((1 - alphas[t]) / torch.sqrt(1 - alpha_hat[t])) * model(x)
            ) + torch.sqrt(betas[t]) * z
            
        return x.cpu().numpy()

    # Generate some samples
    print("Generating images...")
    generated_images = sample_images(16)

    # Get some real images for comparison
    real_images, _ = next(iter(train_loader))
    real_images = real_images[:16].cpu().numpy()

    # Plot results side by side
    fig, axes = plt.subplots(4, 8, figsize=(16, 8))
    fig.suptitle("Real (odd columns) vs Generated (even columns)")
    
    for i in range(4):
        for j in range(4):
            idx = i * 4 + j
            # Denormalize images
            real_img = np.clip((real_images[idx].transpose(1, 2, 0) + 1) / 2, 0, 1)
            gen_img = np.clip((generated_images[idx].transpose(1, 2, 0) + 1) / 2, 0, 1)
            
            axes[i, j * 2].imshow(real_img)
            axes[i, j * 2].axis("off")
            axes[i, j * 2 + 1].imshow(gen_img)
            axes[i, j * 2 + 1].axis("off")
    
    plt.tight_layout()
    plt.show()