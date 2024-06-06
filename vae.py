import argparse
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from torchvision.utils import save_image
from tqdm import tqdm
import librosa
import numpy as np
from PIL import Image
import os
import shutil

# Hyperparameters
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INPUT_DIM = 28 * 28
SOUND_DIM = 2 * 128 * 128  # 2 channels, each with 128x128
HIDDEN_DIM = 200
Z_DIM = 200
NUM_EPOCHS = 1
BATCH_SIZE = 32
LEARNING_RATE = 3e-4

class VariationalAutoEncoder(nn.Module):
    def __init__(self, image_dim, sound_dim, hidden_dim=200, z_dim=20):
        super().__init__()

        # Encoder for image
        self.img_to_hidden = nn.Linear(image_dim, hidden_dim)

        # Encoder for sound
        self.sound_to_hidden = nn.Linear(sound_dim, hidden_dim)

        # Shared latent space
        self.hidden_to_mu = nn.Linear(hidden_dim * 2, z_dim)
        self.hidden_to_sigma = nn.Linear(hidden_dim * 2, z_dim)

        # Decoder
        self.z_to_hidden = nn.Linear(z_dim, hidden_dim)
        self.hidden_to_img = nn.Linear(hidden_dim, image_dim)

        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, img, sound):
        img_hidden = self.relu(self.img_to_hidden(img))
        sound_hidden = self.relu(self.sound_to_hidden(sound))
        sound_hidden = sound_hidden.view(img_hidden.size(0), -1)  # Ensure the dimensions match
        combined_hidden = torch.cat((img_hidden, sound_hidden), dim=1)
        mu = self.hidden_to_mu(combined_hidden)
        sigma = self.hidden_to_sigma(combined_hidden)
        return mu, sigma

    def decode(self, z):
        hidden = self.relu(self.z_to_hidden(z))
        img = self.sigmoid(self.hidden_to_img(hidden))
        return img

    def forward(self, img, sound):
        mu, sigma = self.encode(img, sound)
        epsilon = torch.randn_like(sigma)
        z_reparameterized = mu + sigma * epsilon
        img_reconstructed = self.decode(z_reparameterized)
        return img_reconstructed, mu, sigma

def load_and_transform_audio(file_path):
    y, sr = librosa.load(file_path)
    spectrogram = librosa.feature.melspectrogram(y=y, sr=sr)
    log_spectrogram = librosa.power_to_db(spectrogram, ref=np.max)
    log_spectrogram = librosa.util.fix_length(log_spectrogram, size=128, axis=1)
    return log_spectrogram

class DigitDataset(Dataset):
    def __init__(self, image_paths, sound_file_paths, transform=None):
        self.image_paths = image_paths
        self.sound_file_paths = sound_file_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("L")
        if self.transform:
            img = self.transform(img)
        
        sound_file_path_left = self.sound_file_paths['left'][idx]
        sound_file_path_right = self.sound_file_paths['right'][idx]

        sound_left = load_and_transform_audio(sound_file_path_left)
        sound_right = load_and_transform_audio(sound_file_path_right)

        sound = np.stack((sound_left, sound_right), axis=0)
        sound = torch.tensor(sound, dtype=torch.float32).view(-1)

        return img.view(-1), sound

# Load MNIST dataset and pair images with corresponding audio files
def get_image_paths_and_sound_paths(root, phase):
    image_paths = []
    sound_file_paths = {'left': [], 'right': []}

    mnist_dataset = datasets.MNIST(root=root, train=(phase=='train'), transform=transforms.ToTensor(), download=True)
    mnist_data_loader = DataLoader(dataset=mnist_dataset, batch_size=1, shuffle=False)

    for i, (img, label) in enumerate(mnist_data_loader):
        if i >= len(mnist_dataset):
            break
        image_path = f"{root}/mnist_{phase}_{i}.png"
        transforms.ToPILImage()(img.squeeze()).save(image_path)
        image_paths.append(image_path)
        
        sound_file_paths['left'].append(f"./audio/{phase}/left/left-{label.item()}.wav")
        sound_file_paths['right'].append(f"./audio/{phase}/right/right-{label.item()}.wav")

    return image_paths, sound_file_paths

def train(model, train_loader, optimizer, loss_fn):
    for epoch in range(NUM_EPOCHS):
        loop = tqdm(enumerate(train_loader), total=len(train_loader))
        for i, (img, sound) in loop:
            img, sound = img.to(DEVICE), sound.to(DEVICE)
            img_reconstructed, mu, sigma = model(img, sound)

            reconstruction_loss = loss_fn(img_reconstructed, img)
            kl_div = -0.5 * torch.sum(1 + torch.log(sigma.pow(2)) - mu.pow(2) - sigma.pow(2))

            loss = reconstruction_loss + kl_div
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loop.set_postfix(loss=loss.item())

def inference(model, sound_file_paths, digit, num_examples=1):
    model.eval()
    with torch.no_grad():
        sound_left = load_and_transform_audio(sound_file_paths['left'][digit])
        sound_right = load_and_transform_audio(sound_file_paths['right'][digit])
        sound = np.stack((sound_left, sound_right), axis=0)
        sound = torch.tensor(sound, dtype=torch.float32).view(-1).to(DEVICE)
        
        mu, sigma = model.encode(torch.zeros(1, INPUT_DIM).to(DEVICE), sound.view(1, -1))
        for example in range(num_examples):
            epsilon = torch.randn_like(sigma)
            z = mu + sigma * epsilon
            out = model.decode(z)
            out = out.view(-1, 1, 28, 28)
            save_image(out, f"Generated_{digit}_ex{example}.png")
            print(f"Generated_{digit}_ex{example}.png")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Variational Autoencoder for generating images from audio.')
    parser.add_argument('--train', action='store_true', help='Train the model.')
    parser.add_argument('--inference', action='store_true', help='Run inference to generate images.')

    args = parser.parse_args()

    train_image_paths, train_sound_file_paths = get_image_paths_and_sound_paths("dataset", "train")
    test_image_paths, test_sound_file_paths = get_image_paths_and_sound_paths("dataset", "test")

    train_dataset = DigitDataset(train_image_paths, train_sound_file_paths, transform=transforms.ToTensor())
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Initialize the model, optimizer, and loss function
    model = VariationalAutoEncoder(INPUT_DIM, SOUND_DIM, HIDDEN_DIM, Z_DIM).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCELoss(reduction="sum")

    if args.train:
        train(model, train_loader, optimizer, loss_fn)
        # Save the model after training
        torch.save(model.state_dict(), 'vae_model.pth')

    if args.inference:
        # Load the model for inference
        model.load_state_dict(torch.load('vae_model.pth'))
        output_dir = "generated_images"
        os.makedirs(output_dir, exist_ok=True)

        for idx in range(10):
            inference(model, test_sound_file_paths, idx, num_examples=1)

        for digit in range(10):
            for example in range(1):
                filename = f"Generated_{digit}_ex{example}.png"
                shutil.move(filename, os.path.join(output_dir, filename))
