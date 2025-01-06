class Model(nn.Module):
    def __init__(self, fourier=False, fourier_only=False, random_fourier=False, combined=False, combined_random=False):
        super(Model, self).__init__()
        self.fourier = fourier
        self.random_fourier = random_fourier
        self.fourier_only = fourier_only
        self.combined = combined
        self.combined_random = combined_random

        self.conv_blocks = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
            nn.Dropout(0.3),
        )

        if fourier_only or combined:
            self.fourier_head = nn.Sequential(
                nn.Linear(512 * 8 * 8, 1024),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(1024, 1),
                nn.Sigmoid()
            )

        if not fourier_only:
            self.head = nn.Sequential(
                nn.Linear(512 * 8 * 8, 512),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(512, 1),
                nn.Sigmoid()
            )

        if combined:
            self.combined_head = nn.Sequential(
                nn.Linear(1024, 256),
                nn.ReLU(),
                nn.Dropout(0.3),
                nn.Linear(256, 1),
                nn.Sigmoid()
            )

    def apply_fourier_transform(self, x):
        x = x.float()
        x = torch.fft.fft2(x)
        x_mag = torch.abs(x)
        x_phase = torch.angle(x)
        x_mag_flattened = x_mag.view(x_mag.size(0), -1)
        x_phase_flattened = x_phase.view(x_phase.size(0), -1)
        return x_mag_flattened, x_phase_flattened

    def forward(self, x):
        conv_output = self.conv_blocks(x)
        conv_output_flattened = conv_output.view(conv_output.size(0), -1)

        if self.fourier:
            x_mag_flattened, x_phase_flattened = self.apply_fourier_transform(x)
            combined_features = torch.cat((conv_output_flattened, x_mag_flattened, x_phase_flattened), dim=1)
            return self.head(combined_features)

        elif self.fourier_only:
            x_mag_flattened, x_phase_flattened = self.apply_fourier_transform(x)
            combined_features = torch.cat((x_mag_flattened, x_phase_flattened), dim=1)
            return self.fourier_head(combined_features)

        elif self.combined:
            x_mag_flattened, x_phase_flattened = self.apply_fourier_transform(x)
            combined_features = torch.cat((x_mag_flattened, x_phase_flattened), dim=1)
            x_fourier = self.fourier_head(combined_features)
            return self.combined_head(torch.cat((x_fourier, conv_output_flattened), dim=1))

        else:
            return self.head(conv_output_flattened)