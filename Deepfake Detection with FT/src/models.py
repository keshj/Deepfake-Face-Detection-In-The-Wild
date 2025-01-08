import torch
import torch.nn as nn

'''class Model(nn.Module):
    def __init__(self, use_fourier=False, combined=False):
        super(Model, self).__init__()
        self.use_fourier = use_fourier
        self.combined = combined

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv2d(512, 512, kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # Fourier transform feature head
        if use_fourier:
            self.fourier_head = nn.Sequential(
                # nn.Linear(512 * 8 * 8, 1024),
                nn.Linear(301056, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024 if combined else 512*16*16, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )

    def apply_fourier_transform(self, x):
        """
        Applies Fourier Transform to the input tensor and extracts magnitude and phase.
        """
        x = torch.fft.fft2(x.float())
        x_mag = torch.abs(x)
        x_phase = torch.angle(x)
        x_mag_flattened = x_mag.view(x_mag.size(0), -1)
        x_phase_flattened = x_phase.view(x_phase.size(0), -1)

        # Reduce dimensions using pooling
        return torch.cat((x_mag_flattened, x_phase_flattened), dim=1)

    def forward(self, x):
        # Extract convolutional features
        # print(f"Image shape: {x.shape}")
        conv_output = self.conv_blocks(x)
        # print(f"Conv layer output: {conv_output.shape}")
        conv_output_flattened = conv_output.view(conv_output.size(0), -1)
        # print(f"COnv layer output flattened: {conv_output_flattened.shape}")

        if self.use_fourier:
            # Extract Fourier features
            fourier_features = self.apply_fourier_transform(x)
            # print(fourier_features.shape)
            #fourier_features_flattened = fourier_features.view(fourier_features.size(0), -1)
            fourier_output = self.fourier_head(fourier_features)

            if self.combined:
                # Combine Fourier and convolutional features
                combined_features = torch.cat((conv_output_flattened, fourier_output), dim=1)
                return self.classifier(combined_features)

            # Use only Fourier features
            return self.classifier(fourier_output)
        
        # Use only convolutional features
        return self.classifier(conv_output_flattened)
'''

class FrequencyBranch(nn.Module):
    def __init__(self, output_size=128):
        super(FrequencyBranch, self).__init__()
        # Recalculate input size for 255x255 images
        input_size = 3 * 2 * 128 * 128
        hidden_size1 = 512
        hidden_size2 = 256

        self.fc1 = nn.Linear(input_size, hidden_size1)
        self.fc2 = nn.Linear(hidden_size1, hidden_size2)
        self.fc3 = nn.Linear(hidden_size2, output_size)
        self.relu = nn.ReLU()

    def forward(self, img):
        # Move to CPU for numpy operations
        img_np = img.cpu().numpy()

        features = []
        for channel in range(3):
            f_transform = np.fft.fft2(img_np[channel])
            f_transform_shifted = np.fft.fftshift(f_transform)
            amplitude = np.abs(f_transform_shifted)
            phase = np.angle(f_transform_shifted)
            features.extend([amplitude.flatten(), phase.flatten()])

        input_vector = np.concatenate(features)
        input_tensor = torch.tensor(input_vector, dtype=torch.float32, device=device)

        x = self.relu(self.fc1(input_tensor))
        x = self.relu(self.fc2(x))
        output_vector = self.fc3(x)

        return output_vector
    
class ConvBranch(nn.Module):
    def __init__(self, input_channels=3, output_features=128):
        super(ConvBranch, self).__init__()
        def conv_block(in_channels, out_channels):
            return nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.ReLU(),
                nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(0.3)
            )
        self.model = nn.Sequential(
            conv_block(input_channels, 64),
            conv_block(64, 128),
            conv_block(128, 256),
            conv_block(256, 512),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(512, output_features),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

    def forward(self, image):
        return self.model(image)
    
class CombinedModel(nn.Module):
    def __init__(self):
        super(CombinedModel, self).__init__()
        self.freq_branch = FrequencyBranch(output_size=128)
        self.conv_branch = ConvBranch(output_features=128)
        self.fc1 = nn.Linear(256, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        # Same input for both branches
        freq_output = self.freq_branch(x)
        conv_output = self.conv_branch(x)

        combined = torch.cat((freq_output, conv_output), dim=1)
        x = torch.relu(self.fc1(combined))
        x = self.dropout(x)
        x = torch.relu(self.fc2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.fc3(x))
        return x