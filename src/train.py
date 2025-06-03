import torch
from torch.utils.data import DataLoader
from model import IPAClassifier
from dataset import IPADataset

label_map = {'a': 0, 't': 1, 'e': 2}  # Add full IPA map

dataset = IPADataset('data/labels.csv', 'data/wav', label_map)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

model = IPAClassifier(num_classes=len(label_map))
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
criterion = torch.nn.CrossEntropyLoss()

for epoch in range(10):
    for x, y in loader:
        pred = model(x)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print(f"Epoch {epoch}, Loss: {loss.item()}")

torch.save(model.state_dict(), 'models/model.pth')