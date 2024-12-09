import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from neuronix import accuracy_fn

from tqdm.auto import tqdm  
from timeit import default_timer as timer

import numpy as np
import os 
import matplotlib.pyplot as plt

import sounddevice as sd
from scipy.io.wavfile import write
import librosa
import librosa.display


# Set device to GPU if available, otherwise use CPU
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class CNN_Model(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int, output_shape:int):
        super().__init__()

        # First block with hidden_units filters
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),  # hidden_units as number of filters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Second block with hidden_units * 2 filters
        self.block_2 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),  # hidden_units * 2 filters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 2, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Third block with hidden_units * 4 filters
        self.block_3 = nn.Sequential(
            nn.Conv2d(in_channels=hidden_units * 2, out_channels=hidden_units * 4, kernel_size=3, stride=1, padding=1),  # hidden_units * 4 filters
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units * 4, out_channels=hidden_units * 4, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        # Classifier with flattened output
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 4 * 3 * 3, 512),  # Adjust based on the output size after pooling
            nn.ReLU(),
            nn.Linear(512, output_shape)
        )

    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        x = self.block_2(x)
        x = self.block_3(x)
        x = self.classifier(x)
        return x


# Function to print training time
def print_train_time(start: float, end: float, device: torch.device = None):
    total_time = end - start  # Calculate total training time
    print(f"Train time on {device}: {total_time:.3f} seconds")  # Print training time
    return total_time


# Function to perform a single training step
def train_step(model: torch.nn.Module,
               data_loader: torch.utils.data.DataLoader,
               loss_fn: torch.nn.Module,
               optimizer: torch.optim.Optimizer,
               accuracy_fn,
               device: torch.device = device):
    train_loss, train_acc = 0, 0  # Initialize loss and accuracy
    model.to(device)  # Move model to device
    for batch, (X, y) in enumerate(data_loader):
        X, y = X.to(device), y.to(device)  # Move batch data to device

        y_pred = model(X)  # Forward pass: compute predictions

        loss = loss_fn(y_pred, y)  # Compute loss
        train_loss += loss  # Accumulate training loss
        train_acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Compute accuracy

        optimizer.zero_grad()  # Zero gradients from previous step
        loss.backward()  # Backward pass: compute gradients
        optimizer.step()  # Update model parameters

    # Average the training loss and accuracy over the number of batches
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss: {train_loss:.5f} | Train accuracy: {train_acc:.2f}%")  # Print training metrics


# Function to perform a single testing step
def test_step(data_loader: torch.utils.data.DataLoader,
              model: torch.nn.Module,
              loss_fn: torch.nn.Module,
              accuracy_fn,
              device: torch.device = device):
    test_loss, test_acc = 0, 0  # Initialize test loss and accuracy
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # Move batch data to device
            
            test_pred = model(X)  # Forward pass: compute predictions
            
            test_loss += loss_fn(test_pred, y)  # Compute test loss
            test_acc += accuracy_fn(y_true=y, y_pred=test_pred.argmax(dim=1))  # Compute accuracy
        
        # Average the test loss and accuracy over the number of batches
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"Test loss: {test_loss:.5f} | Test accuracy: {test_acc:.2f}%\n")  # Print testing metrics


# Function to evaluate the model on a dataset
def eval_model(model: torch.nn.Module, 
               data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, 
               accuracy_fn,
               device: torch.device = device):

    loss, acc = 0, 0  # Initialize total loss and accuracy
    model.to(device)  # Move model to device
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for X, y in data_loader:
            X, y = X.to(device), y.to(device)  # Move batch data to device
            
            y_pred = model(X)  # Forward pass: compute predictions
            
            loss += loss_fn(y_pred, y).item()  # Accumulate loss
            acc += accuracy_fn(y_true=y, y_pred=y_pred.argmax(dim=1))  # Accumulate accuracy
        
        # Average the loss and accuracy over the number of batches
        loss /= len(data_loader)
        acc /= len(data_loader)
    
    return {"model_name": model.__class__.__name__,
            "model_loss": loss,
            "model_acc": acc}  # Return evaluation metrics


# Function to make predictions on a list of samples
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []  # List to hold prediction probabilities
    model.eval()  # Set model to evaluation mode
    with torch.inference_mode():  # Disable gradient tracking
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)  # Add batch dimension and move to device

            pred_logit = model(sample)  # Forward pass: compute logits

            # Apply softmax to get probabilities
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)  # Perform softmax on the logits dimension

            pred_probs.append(pred_prob.cpu())  # Append predictions to the list
            
    return torch.stack(pred_probs)  # Return stacked probabilities


# Define transformations
transform = transforms.Compose([
    transforms.Resize((28, 28)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ToTensor(),
])

# Load the spectrogram dataset
train_data = datasets.ImageFolder(root="datasets", transform=transform)
test_data = train_data

# Set batch size for DataLoader
BATCH_SIZE = 4
train_dataloader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)  # Shuffle training data
test_dataloader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)  # No shuffle for test data
class_names = train_data.classes  # Get class names for labeling predictions

# Initialize the CNN model
model_2 = CNN_Model(input_shape=3, hidden_units=64, output_shape=len(class_names)).to(device)

# Define loss function and optimizer
loss_fn = nn.CrossEntropyLoss()  # For multi-class classification
optimizer = torch.optim.SGD(params=model_2.parameters(), lr=0.001)  # Stochastic Gradient Descent

# Start timer for training duration
train_time_start_model_2 = timer()

# Set number of epochs
epochs = 15
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")  # Print current epoch
    # Train the model for one epoch
    train_step(data_loader=train_dataloader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    # Test the model after training
    test_step(data_loader=test_dataloader,
        model=model_2,
        loss_fn=loss_fn,
        accuracy_fn=accuracy_fn,
        device=device
    )

# End timer for training duration
train_time_end_model_2 = timer()
# Print total training time
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)

# Evaluate the model on the test dataset
model_2_results = eval_model(
    model=model_2,
    data_loader=test_dataloader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
print(model_2_results)  # Print evaluation results


# Directory paths for saving audio and spectrograms
use_dir = "buffer"
os.makedirs(use_dir, exist_ok=True)

# Set audio recording parameters
sample_rate = 16000
duration = 5
channels = 1 

# Record audio
print(f"Recording for {duration} seconds... Speak now!")
audio = sd.rec(int(duration * sample_rate), samplerate=sample_rate, channels=channels, dtype='int16')
sd.wait()

# Save the audio file
audio_filename = os.path.join(use_dir, "recording.wav")
write(audio_filename, sample_rate, audio)
print(f"Audio saved to {audio_filename}")

# Generate spectrogram
y, sr = librosa.load(audio_filename, sr=sample_rate)  # Load audio at the specified sample rate
S = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
S_db = librosa.amplitude_to_db(S, ref=np.max)

# Plot and save the spectrogram
plt.figure(figsize=(10, 6))
librosa.display.specshow(S_db, sr=sr, hop_length=512, cmap='viridis')
plt.axis('off')  # Hide axes

spectrogram_filename = os.path.join(use_dir, "spectrogram.png")
plt.savefig(spectrogram_filename, dpi=300, bbox_inches='tight', pad_inches=0)
plt.close()
print(f"Spectrogram saved to {spectrogram_filename}")

# Prepare spectrogram for model prediction
S_db_resized = np.resize(S_db, (3, 28, 28))
S_db_tensor = torch.tensor(S_db_resized, dtype=torch.float32).to(device)

# Make predictions using the trained model
model_2.eval()
with torch.no_grad():
    pred_probs = make_predictions(model_2, [S_db_tensor], device=device)

# Print prediction probabilities
pred_class_idx = pred_probs.argmax(dim=1).item()
pred_class = class_names[pred_class_idx]
print(f"Predicted class: {pred_class} | Prediction probabilities: {pred_probs}")

# Save the model
torch.save(model_2.state_dict(), 'speaker_recognition.pth')
