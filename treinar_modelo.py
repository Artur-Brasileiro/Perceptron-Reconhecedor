import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from sklearn.metrics import classification_report, accuracy_score

# Configura o dispositivo (CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Usando dispositivo: {device}")

# Transformações
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))
])

# Carrega os dados
train_dataset = datasets.EMNIST(root='data', split='letters', train=True, download=True, transform=transform)
test_dataset = datasets.EMNIST(root='data', split='letters', train=False, download=True, transform=transform)

# Converte para problema binário (A vs Não-A)
train_targets = torch.where(train_dataset.targets == 1, 1, 0)
test_targets = torch.where(test_dataset.targets == 1, 1, 0)
train_dataset.targets = train_targets
test_dataset.targets = test_targets

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

# --- Modelo Perceptron ---
class Perceptron(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(28*28, 1)
        
    def forward(self, x):
        return self.fc(x)

model = Perceptron().to(device)

# --- CORREÇÃO DO DESEQUILÍBRIO (v5.0) ---
pos_weight = torch.tensor([4.0]).to(device) 

criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight) 
optimizer = optim.SGD(model.parameters(), lr=0.1)

print("Iniciando o treinamento (v5.0 - Mais Calmo)...")
for epoch in range(30):
    model.train()
    total_loss = 0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.float().to(device)
        
        optimizer.zero_grad()
        outputs = model(images).squeeze() 
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    print(f"Época {epoch+1}/30 - Loss: {total_loss/len(train_loader):.4f}")

print("Treinamento concluído.")

# --- Avaliação ---
model.eval()
y_true, y_pred = [], []

with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.float().to(device)
        outputs = torch.sigmoid(model(images)).squeeze() 
        preds = (outputs > 0.5).float()
        y_true.extend(labels.cpu().numpy())
        y_pred.extend(preds.cpu().numpy())

acc = accuracy_score(y_true, y_pred)
print(f"\nPrecisão final no set de teste (v5.0): {acc*100:.2f}%")
print(classification_report(y_true, y_pred, target_names=["Não-A", "A"]))

# Salva o novo cérebro v5.0
torch.save(model.state_dict(), 'perceptron_A_v5.pth')
print("\nModelo treinado (v5.0) salvo como 'perceptron_A_v5.pth'")