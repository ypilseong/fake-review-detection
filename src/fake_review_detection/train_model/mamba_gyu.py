import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel
from mamba_ssm import Mamba
from sklearn.model_selection import train_test_split
from tqdm import tqdm

# 데이터 로드
with open('data/final_data_update3.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터 전처리
# texts = [item['textual']['content'] for item in data['store']]
# labels = [item['labels'][0] for item in data['store']]
texts = []
labels = []
for store in data['store']:
    if 'textual' in store and 'content' in store['textual'] and 'labels' in store:
        texts.append(store['textual']['content'])
        labels.append(2 if store['labels'] == -1 else store['labels'])

# BERT 토크나이저 및 모델 로드
tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
bert_model = BertModel.from_pretrained('bert-base-multilingual-cased')

# 커스텀 데이터셋 클래스
class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]

        encoding = self.tokenizer.encode_plus(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'label': torch.tensor(label, dtype=torch.long)
        }

# 데이터 분할
train_texts, val_texts, train_labels, val_labels = train_test_split(texts, labels, test_size=0.2, random_state=42)

# 데이터셋 및 데이터로더 생성
max_length = 512
batch_size = 8

train_dataset = TextDataset(train_texts, train_labels, tokenizer, max_length)
val_dataset = TextDataset(val_texts, val_labels, tokenizer, max_length)

train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size)

# Mamba 모델 정의
class MambaClassifier(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.mamba = Mamba(
            d_model=input_dim,
            d_state=16,
            d_conv=4,
            expand=2
        )
        self.fc = torch.nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.mamba(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.fc(x)
        return x

# BERT와 Mamba를 결합한 모델
class BERTMambaClassifier(torch.nn.Module):
    def __init__(self, bert_model, mamba_model):
        super().__init__()
        self.bert = bert_model
        self.mamba = mamba_model

    def forward(self, input_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, attention_mask=attention_mask)[0]
        output = self.mamba(bert_output)
        return output

# 모델 초기화
mamba_model = MambaClassifier(768, 768, 3)  # 3 classes: -1, 0, 1
model = BERTMambaClassifier(bert_model, mamba_model)

# 학습 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)

# 학습 함수
def train(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    for batch in tqdm(dataloader, desc="Training"):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['label'].to(device).view(-1)

        optimizer.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

# 평가 함수
def evaluate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['label'].to(device).view(-1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    return total_loss / len(dataloader), accuracy

# 학습 루프
num_epochs = 100
for epoch in range(num_epochs):
    train_loss = train(model, train_dataloader, criterion, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_dataloader, criterion, device)
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")

# 모델 저장
torch.save(model.state_dict(), 'bert_mamba_classifier.pth')

print("Training completed and model saved.")

