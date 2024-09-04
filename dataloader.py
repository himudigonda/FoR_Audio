import os
import torch
import torchaudio
from torch.utils.data import Dataset, DataLoader

class AudioDataset(Dataset):
    def __init__(self, root_dir, sample_rate=16000, duration=5):
        self.root_dir = root_dir
        self.sample_rate = sample_rate
        self.duration = duration
        self.files = []
        self.labels = []

        for label, folder in enumerate(['real', 'fake']):
            folder_path = os.path.join(root_dir, folder)
            for file in os.listdir(folder_path):
                if file.endswith('.wav'):
                    self.files.append(os.path.join(folder_path, file))
                    self.labels.append(label)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        audio_path = self.files[idx]
        waveform, sr = torchaudio.load(audio_path)

        if sr != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

        target_length = self.duration * self.sample_rate
        if waveform.size(1) > target_length:
            waveform = waveform[:, :target_length]
        else:
            waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))

        mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=self.sample_rate)(waveform)

        return mel_spec, self.labels[idx]

def get_dataloader(root_dir, batch_size=32, shuffle=True):
    dataset = AudioDataset(root_dir)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
