import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, use_fourier=True, combined=True):
        super(Model, self).__init__()
        self.use_fourier = use_fourier
        self.combined = combined

        # Convolutional feature extractor
        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),

            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3)
        )

        # Fourier transform feature head
        if use_fourier:
            self.fourier_head = nn.Sequential(
                nn.Linear(512 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 512),
                nn.ReLU(),
                nn.Dropout(0.3)
            )

        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(1024 if combined else 512, 256),
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
        conv_output = self.conv_blocks(x)
        conv_output_flattened = conv_output.view(conv_output.size(0), -1)

        if self.use_fourier:
            # Extract Fourier features
            fourier_features = self.apply_fourier_transform(x)
            fourier_output = self.fourier_head(fourier_features)

            if self.combined:
                # Combine Fourier and convolutional features
                combined_features = torch.cat((conv_output_flattened, fourier_output), dim=1)
                return self.classifier(combined_features)

            # Use only Fourier features
            return self.classifier(fourier_output)
        
        # Use only convolutional features
        return self.classifier(conv_output_flattened)
