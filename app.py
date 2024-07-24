import json
import os
import boto3
import torch
from transformers import BertModel
from kobert_tokenizer import KoBERTTokenizer
from torch import nn


# 환경 변수 설정
os.environ['HF_HOME'] = '/tmp'

s3 = boto3.client('s3')
bucket_name = 'sorktjrrb-aws-sentiment-analysis'
model_key = 'kobert_model_epoch_5.pth'
model_path = '/tmp/kobert_model_epoch_5.pth'

# KoBERT 모델 및 토크나이저 불러오기
tokenizer = KoBERTTokenizer.from_pretrained('skt/kobert-base-v1')
bertmodel = BertModel.from_pretrained('skt/kobert-base-v1')
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    try:
        print(f"Attempting to download model from S3 bucket: {bucket_name}, key: {model_key}")
        s3.download_file(bucket_name, model_key, model_path)
        if os.path.exists(model_path):
            print("모델 다운로드 성공")
            return model_path
        else:
            print("모델 파일이 존재하지 않습니다.")
            raise FileNotFoundError("Downloaded model file not found.")
    except Exception as e:
        print(f"Error downloading model: {e}")
        raise RuntimeError(f"Error downloading model: {e}")

# BERTClassifier 클래스 정의
class EmotionClassifier(nn.Module):
    def __init__(self, bert, hidden_size=768, num_classes=7, dr_rate=0.8):  # 드롭아웃 비율을 0.8로 설정
        super(EmotionClassifier, self).__init__()
        self.bert = bert
        self.dr_rate = dr_rate
        self.classifier = nn.Linear(hidden_size, num_classes)
        if dr_rate:
            self.dropout = nn.Dropout(p=dr_rate)
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.batch_norm = nn.BatchNorm1d(hidden_size)  # Batch Normalization 추가

    def gen_attention_mask(self, token_ids, valid_length):
        attention_mask = torch.zeros_like(token_ids)
        for i, v in enumerate(valid_length):
            attention_mask[i][:int(v)] = 1
        return attention_mask.float()

    def forward(self, token_ids, valid_length, segment_ids):
        attention_mask = self.gen_attention_mask(token_ids, valid_length)
        outputs = self.bert(input_ids=token_ids, token_type_ids=segment_ids, attention_mask=attention_mask)
        pooler = outputs[1]
        if self.dr_rate:
            out = self.dropout(pooler)
        else:
            out = pooler
        out = self.layer_norm(out)
        out = self.batch_norm(out)  # Batch Normalization 적용
        return self.classifier(out)

# 모델 로드
model_file_path = load_model()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EmotionClassifier(bertmodel, dr_rate=0.5).to(device)  # dr_rate를 저장된 모델의 값으로 설정
with open(model_file_path, 'rb') as f:
    state_dict = torch.load(f, map_location=torch.device('cpu'))
model.load_state_dict(state_dict)
model.eval()
label_map = {
    "fear": 0,
    "surprise": 1,
    "angry": 2,
    "sadness": 3,
    "neutral": 4,
    "happiness": 5,
    "disgust": 6
}
reverse_label_map = {v: k for k, v in label_map.items()}

def preprocess(input_text):
    encoding = tokenizer.encode_plus(
        input_text,
        max_length=10,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return encoding['input_ids'], encoding['token_type_ids'], encoding['attention_mask']


# 예측 함수
def predict(sentence, model, tokenizer, device, max_len=128):
    model.eval()
    encoding = tokenizer.encode_plus(
        sentence,
        add_special_tokens=True,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    input_ids = encoding['input_ids'].to(device)
    attention_mask = encoding['attention_mask'].to(device)
    token_type_ids = encoding['token_type_ids'].to(device)
    valid_length = (input_ids != tokenizer.pad_token_id).sum(dim=1)

    with torch.no_grad():
        outputs = model(input_ids, valid_length, token_type_ids)
        predicted_class = torch.argmax(outputs, dim=1).item()  # .item()을 사용하여 스칼라 값을 얻음

    return outputs  # logits를 반환하도록 수정됨

def lambda_handler(event, context):
    try:
        print("Lambda 함수가 시작되었습니다.")
        body = event['body']
        # JSON 파싱 시도
        try:
            if isinstance(body, bytes):
                body = body.decode('utf-8')
            data = json.loads(body)
        except json.JSONDecodeError:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Invalid JSON format'})
            }
        input_text = data.get('input')
        if not input_text:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': 'Missing "input" in request body'})
            }
        input_ids, token_type_ids, attention_mask = preprocess(input_text)
        # 여기서 device 인수 추가
        logits = predict(input_text, model, tokenizer, device)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        emotion_probabilities = {reverse_label_map[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(probabilities))}
        return {
            'statusCode': 200,
            'body': json.dumps({'logits': logits.numpy().tolist(), 'prediction': emotion_probabilities})
        }
    except Exception as e:
        print(f"예외 발생: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }