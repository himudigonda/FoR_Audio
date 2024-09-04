import torch
import torchaudio
from models import AudioClassifier
from dataloader import get_dataloader
from utils import load_model
from metrics import accuracy, confusion_matrix

def predict(audio_path, model, device, sample_rate=16000, duration=5):
    model.eval()
    waveform, sr = torchaudio.load(audio_path)

    if sr != sample_rate:
        waveform = torchaudio.functional.resample(waveform, sr, sample_rate)

    target_length = duration * sample_rate
    if waveform.size(1) > target_length:
        waveform = waveform[:, :target_length]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, target_length - waveform.size(1)))

    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=sample_rate)(waveform)
    mel_spec = mel_spec.unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(mel_spec)
        _, predicted = torch.max(output, 1)

    return "Real" if predicted.item() == 0 else "Fake"

def evaluate_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = load_model(AudioClassifier(), 'model.pth').to(device)
    model.eval()

    test_loader = get_dataloader('data/test', batch_size=32, shuffle=False)

    all_predictions = []
    all_labels = []

    for inputs, labels in test_loader:
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.no_grad():
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
        all_predictions.extend(predicted.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

    test_accuracy = accuracy(all_predictions, all_labels)
    conf_matrix = confusion_matrix(all_predictions, all_labels)

    print(f'Test Accuracy: {test_accuracy:.4f}')
    print('Confusion Matrix:')
    print(conf_matrix)
