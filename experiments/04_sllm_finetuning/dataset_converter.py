import json
import pandas as pd

# JSON 데이터 파일 로드
input_file = 'datasets/baemin_dataset.json'  # JSON 파일 경로
output_file = 'datasets/baemin_dataset.csv'  # 저장할 CSV 파일 경로

# JSON 데이터 읽기
with open(input_file, 'r', encoding='utf-8') as f:
    data = json.load(f)

# 데이터를 분리하는 함수 정의
def parse_text(data):
    rows = []
    for item in data:
        text = item['text']
        
        # <s>[INST]와 [/INST]를 기준으로 나누기
        if "<s>[INST]" in text and "[/INST]" in text:
            instruction_start = text.find("<s>[INST] Q.") + len("<s>[INST] Q.")
            instruction_end = text.find("[/INST]")
            instruction = text[instruction_start:instruction_end].strip()
            
            output = text[instruction_end + len("[/INST]"):].replace("</s>", "").strip()
            
            rows.append({
                "instruction": instruction,
                "input": "",  # input이 없으므로 빈 값
                "output": output
            })
    return rows

# JSON 데이터 파싱
parsed_data = parse_text(data)

# DataFrame 생성
df = pd.DataFrame(parsed_data)

# CSV 파일로 저장
df.to_csv(output_file, index=False, encoding="utf-8-sig")