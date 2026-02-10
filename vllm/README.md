# vLLM 서빙 구조

모델은 `/home/h202403659/LLM-Server/models/llm` 아래에 있는 것을 **이름만 바꿔서** 쉽게 전환할 수 있습니다.

## 디렉토리 구성

- `configs/`: 환경 변수 파일
- `scripts/`: 서빙 실행 스크립트
- `logs/`: 로그 출력 폴더

## 모델 변경 방법

`configs/vllm.env`의 `MODEL_NAME`만 변경하면 됩니다.

예시:
```
MODEL_NAME=gemma3-12b-it
```

## 실행

```
cd /home/h202403659/LLM-Server/serving/vllm
./scripts/serve.sh
```
