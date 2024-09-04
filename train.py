import torch
import torch.nn as nn
import torch.optim as optim
from models import AudioClassifier
from dataloader import get_dataloader
from utils import save_model
from metrics import accuracy

def train_model(num_epochs=10, learning_rate=0.001, batch_size=32):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AudioClassifier().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    train_loader = get_dataloader('data/train', batch_size=batch_size)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        running_acc = 0.0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            running_acc += accuracy(predicted, labels)

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = running_acc / len(train_loader)
        print(f'Epoch {epoch+1}, Loss: {epoch_loss:.4f}, Accuracy: {epoch_acc:.4f}')

    save_model(model, 'model.pth')
    print('Finished Training')
