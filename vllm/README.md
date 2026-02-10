
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
./serve.sh
```
