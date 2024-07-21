import json
from collections import Counter

# JSON 데이터
with open('data/final_data_update.json', 'r', encoding='utf-8') as file:
    data = json.load(file)

# JSON 데이터를 파싱

# labels 값을 추출
labels = [store["labels"] for store in data["store"]]

# 리스트를 평탄화
flat_labels = [label for sublist in labels for label in sublist]

# 각 값의 개수를 계산
label_counts = Counter(flat_labels)

# 결과 출력
print(label_counts)
