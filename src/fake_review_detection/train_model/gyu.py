import json

# 원본 JSON 파일 경로
input_file_path = 'data/final_data_update2.json'

# 변경된 JSON 파일 저장 경로
output_file_path = 'data/final_data_update3.json'

# JSON 파일 읽기
with open(input_file_path, 'r', encoding='utf-8') as file:
    data = json.load(file)

# labels 값이 -1인 경우 2로 변경
for store in data['store']:
    store['labels'] = [2 if label == -1 else label for label in store['labels']]

# 변경된 JSON 데이터를 새로운 파일에 쓰기
with open(output_file_path, 'w', encoding='utf-8') as file:
    json.dump(data, file, ensure_ascii=False, indent=4)

print("Labels 값이 -1인 경우 2로 변경 완료 및 data/final_data_update3.json으로 저장 완료")
