import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from transformers import BertTokenizer, BertModel
from mamba_ssm import Mamba
from tqdm import tqdm


# CUDA 장치 설정
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 1. 데이터 준비
# 텍스트 데이터를 임베딩 벡터로 변환
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
bert_model = BertModel.from_pretrained('bert-base-uncased')

with open('data/final_data_update2.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# 데이터에서 텍스트와 레이블 추출
content_list = []
labels_list = []
for store in data['store']:
    if 'textual' in store and 'content' in store['textual'] and 'labels' in store:
        content_list.append(store['textual']['content'])
        labels_list.append(2 if store['labels'] == -1 else store['labels'])  # -1 값을 2로 변경

# 텍스트 데이터 전처리
inputs = tokenizer(content_list, return_tensors="pt", padding=True, truncation=True)
outputs = bert_model(**inputs)
embedding = outputs.last_hidden_state  # (batch_size, seq_len, hidden_dim)

# GPU로 이동
embedding = embedding.to(device)

# 2. 모델 초기화
embedding_dim = 768  # BERT 모델의 임베딩 차원

# Mamba 모델
mamba_model = Mamba(d_model=embedding_dim, d_state=16, d_conv=4, expand=2).to(device)

# 분류기
classifier = nn.Linear(embedding_dim, 3).to(device)  # 3-class 분류 예시

# 손실 함수와 옵티마이저
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(list(mamba_model.parameters()) + list(classifier.parameters()), lr=0.001)

# 3. 출력 벡터를 분류기에 입력
def forward_pass(inputs):
    # 텍스트 임베딩
    mamba_output = mamba_model(inputs)
    final_vector = mamba_output[:, -1, :]  # (batch_size, d_model)
    return final_vector

# 4. 학습 루프
num_epochs = 10

# 레이블 텐서로 변환
labels = torch.tensor(labels_list, dtype=torch.long, device=device).view(-1)  # 레이블이 정수형인지 확인

# 데이터셋 및 데이터로더 생성
dataset = TensorDataset(embedding, labels)
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# 데이터셋 및 레이블 차원 확인
print("Embedding shape:", embedding.shape)
print("Labels shape:", labels.shape)
print("Sample labels:", labels[:10])

# 레이블 값의 범위 확인
num_classes = 3  # 클래스 수 정의
invalid_labels = labels[(labels < 0) | (labels >= num_classes)]
if len(invalid_labels) > 0:
    print(f"Invalid labels found: {invalid_labels}")
else:
    print("All labels are valid.")

for epoch in range(num_epochs):
    mamba_model.train()
    classifier.train()
    
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}", unit="batch"):
        optimizer.zero_grad()
        combined_vector = forward_pass(inputs.to(device))
        outputs = classifier(combined_vector)
        loss = criterion(outputs, labels.to(device))
        loss.backward(retain_graph=True) 
        optimizer.step()

        running_loss += loss.item()

        # 메모리 캐시 비우기
        
    
    avg_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs}, Loss: {avg_loss:.4f}")

print(labels_list[:10])
