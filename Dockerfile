# 베이스 이미지 설정
FROM public.ecr.aws/lambda/python:3.10

# 작업 디렉토리 설정
WORKDIR /var/task

# 필요한 패키지 설치
RUN yum -y install gcc-c++ make ca-certificates git

# requirements.txt 복사 및 패키지 설치
COPY requirements.txt .
RUN pip3 install --trusted-host pypi.org --trusted-host files.pythonhosted.org --no-cache-dir -r requirements.txt

# 애플리케이션 소스 코드 복사
COPY . .

# 애플리케이션 실행 명령어 설정
CMD ["app.lambda_handler"]