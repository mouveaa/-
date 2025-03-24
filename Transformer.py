# Trying out a transformer on MNIST - not sure if this is overkill but let's see...
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F  # might need this later
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# TODO: Try running this on GPU if available, otherwise fallback to CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Playing around with these values - might need adjustment later
BATCH_SIZE = 64  # seems reasonable for my GPU memory
LEARNING_RATE = 1e-4  # using the "magic" learning rate from Karpathy's tweet
EPOCHS = 10  # probably need more but let's start here
NUM_CLASSES = 10  # MNIST stuff
EMBED_DIM = 128    # not sure if this is enough but worth a shot
NUM_HEADS = 8     # standard transformer head count
NUM_LAYERS = 6    # keeping it shallow for now
FFN_HIDDEN_DIM = 256  # might be too small?
DROPOUT = 0.05     # default dropout, fingers crossed

# Basic MNIST preprocessing - keeping it simple for now
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # these magic numbers are from somewhere on StackOverflow
])

# Get that MNIST data
train_dataset = datasets.MNIST('./data', train=True, transform=transform, download=True)
test_dataset = datasets.MNIST('./data', train=False, transform=transform, download=True)

# Note to self: maybe try different batch sizes if training is slow
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, num_classes, ffn_hidden_dim, dropout):
        super().__init__()  # shorter version of super call
        
        # Linear projection instead of embedding - not sure if this is the best way
        self.embedding = nn.Linear(input_dim, embed_dim)
        
        # Random initialization - maybe should use sin/cos encoding like the paper?
        self.pos_encoding = nn.Parameter(torch.randn(1, 28, embed_dim))
        
        # Stacking transformer layers - hope this works
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, 
            nhead=num_heads,
            dim_feedforward=ffn_hidden_dim,
            dropout=dropout,
            batch_first=True  # always forget this parameter
        )
        
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        self.classifier = nn.Linear(embed_dim, num_classes)  # final classification head

    def forward(self, x):
        # Add positional encoding - probably not the best way but it works
        x = self.embedding(x) + self.pos_encoding
        
        # Run through transformer - fingers crossed
        features = self.transformer(x)
        
        # Average pooling - maybe max pooling would be better?
        pooled = features.mean(dim=1)
        
        return self.classifier(pooled)

# Initialize our monster
model = TransformerClassifier(
    input_dim=28,  # MNIST image width
    embed_dim=EMBED_DIM,
    num_heads=NUM_HEADS,
    num_layers=NUM_LAYERS,
    num_classes=NUM_CLASSES,
    ffn_hidden_dim=FFN_HIDDEN_DIM,
    dropout=DROPOUT
).to(device)

# Basic stuff - might try different optimizer later
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Keep track of everything - will plot these later
history = {
    'train_loss': [],
    'test_loss': [],
    'train_acc': [],
    'test_acc': []
}

def train_epoch():
    model.train()
    correct = total = running_loss = 0
    
    for batch_idx, (imgs, labels) in enumerate(train_loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        
        # Reshape images for transformer input
        batch_size = imgs.size(0)
        imgs = imgs.view(batch_size, 28, 28)
        
        # Standard training loop
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy - probably a better way to do this
        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += labels.size(0)
        correct += (preds == labels).sum().item()
        
        if batch_idx % 100 == 0:  # print something so I know it's working
            print(f'Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}')
    
    return running_loss / len(train_loader), 100 * correct / total

def evaluate():
    model.eval()
    correct = total = running_loss = 0
    
    with torch.no_grad():  # don't forget this!
        for imgs, labels in test_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)
            imgs = imgs.view(-1, 28, 28)
            
            outputs = model(imgs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            
            _, preds = outputs.max(1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
    
    return running_loss / len(test_loader), 100 * correct / total

# Training loop - let's see if this works
for epoch in range(EPOCHS):
    print(f'\nEpoch {epoch+1}/{EPOCHS}')
    print('-' * 20)
    
    train_loss, train_acc = train_epoch()
    test_loss, test_acc = evaluate()
    
    # Save metrics
    history['train_loss'].append(train_loss)
    history['test_loss'].append(test_loss)
    history['train_acc'].append(train_acc)
    history['test_acc'].append(test_acc)
    
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.2f}%')

# Quick visualization
plt.figure(figsize=(12, 4))

plt.subplot(1, 2, 1)
plt.plot(history['train_loss'], label='Train')
plt.plot(history['test_loss'], label='Test')
plt.title('Loss Over Time')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history['train_acc'], label='Train')
plt.plot(history['test_acc'], label='Test')
plt.title('Accuracy Over Time')
plt.legend()

plt.tight_layout()
plt.show()