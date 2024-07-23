import json
import boto3
import torch
import os
from transformers import BertModel, BertTokenizer

# 환경 변수 설정
os.environ['HF_HOME'] = '/tmp'

s3 = boto3.client('s3')
bucket_name = 'sorktjrrb-aws-sentiment-analysis'
model_key = 'kobert_model_epoch_5.pth'
model_path = '/tmp/kobert_model_epoch_5.pth'

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
        raise RuntimeError(f"Error downloading model: {e}")

class EmotionClassifier(torch.nn.Module):
    def __init__(self, num_classes):
        super(EmotionClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('monologg/kobert')
        self.dropout = torch.nn.Dropout(0.3)
        self.classifier = torch.nn.Linear(768, num_classes)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None):
        outputs = self.bert(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        pooled_output = outputs[1]
        dropout_output = self.dropout(pooled_output)
        logits = self.classifier(dropout_output)
        return logits

# 모델 로드
model_file_path = load_model()
model = EmotionClassifier(num_classes=7)
state_dict = torch.load(model_file_path, map_location=torch.device('cpu'))
model.load_state_dict(state_dict, strict=False)  # strict=False 옵션 추가
model.eval()

tokenizer = BertTokenizer.from_pretrained('monologg/kobert')

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

def predict(input_ids, token_type_ids, attention_mask):
    with torch.no_grad():
        logits = model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
    return logits

def lambda_handler(event, context):
    try:
        body = event['body']
        if isinstance(body, bytes):
            body = body.decode('utf-8')
        data = json.loads(body)
        input_text = data['input']

        input_ids, token_type_ids, attention_mask = preprocess(input_text)
        logits = predict(input_ids, token_type_ids, attention_mask)
        probabilities = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]
        emotion_probabilities = {reverse_label_map[i]: round(float(probabilities[i]) * 100, 2) for i in range(len(probabilities))}
        return {
            'statusCode': 200,
            'body': json.dumps({'logits': logits.numpy().tolist(), 'prediction': emotion_probabilities})
        }
    except Exception as e:
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }