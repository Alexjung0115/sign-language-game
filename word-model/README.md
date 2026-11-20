# Word Sign AI Server

19개 단어 수어를 인식하는 CPU PyTorch WebSocket 서비스다. 브라우저가
MediaPipe raw 양손 랜드마크와 원본 영상 크기를 `/word`로 보내면 서버가
픽셀 좌표 복원, 20~60프레임 전처리, 추론 및 확률 smoothing을 수행한다.

## 연동·배포 문서

- 프론트 전달용 최종 명세: [`docs/frontend-integration-spec.md`](docs/frontend-integration-spec.md)
- EC2 배포 체크리스트: [`docs/ec2-deployment-checklist.md`](docs/ec2-deployment-checklist.md)

## 모델 파일

체크포인트는 Git 저장소에서 버전을 관리한다. GitLab 배포 작업이 체크섬을
검증한 뒤 EC2 호스트의 다음 경로에 자동 설치하며, Docker 이미지에는 중복으로
포함하지 않는다.

```text
/opt/sudal/ai/models/word-19-v1/best_model.pt
```

컨테이너 기본 경로:

```text
/app/models/word-19-v1/best_model.pt
```

선택적으로 `HANDPRACTICE_WORD_CHECKPOINT_SHA256`을 설정하면 시작할 때
체크포인트 checksum을 검증한다.

수동 복구가 필요한 경우의 EC2 배치 예시:

```bash
sudo install -d -m 755 /opt/sudal/ai/models
sudo install -d -m 755 /opt/sudal/ai/models/word-19-v1
sudo chmod 755 \
  /opt/sudal/ai/models \
  /opt/sudal/ai/models/word-19-v1
sudo install -m 644 best_model.pt \
  /opt/sudal/ai/models/word-19-v1/best_model.pt
echo "898169cf56bfbe39a3ce3b13165a8d6a53c5b873033b278ad0ab1ca7f17a72f4  /opt/sudal/ai/models/word-19-v1/best_model.pt" \
  | sha256sum -c -
```

## 실행

```bash
python -m server.main
```

환경 변수:

| 이름 | 기본값 |
|---|---|
| `HANDPRACTICE_WORD_HOST` | `localhost` |
| `HANDPRACTICE_WORD_PORT` | `8767` |
| `HANDPRACTICE_WORD_PATH` | `/word` |
| `HANDPRACTICE_WORD_CHECKPOINT` | `models/word-19-v1/best_model.pt` |
| `HANDPRACTICE_WORD_TORCH_THREADS` | `2` |
| `HANDPRACTICE_WORD_INFERENCE_WORKERS` | `1` |

## 프로토콜 핵심

- `GET_CAPABILITIES`
- `LANDMARK_FRAME`: `hands.left/right`, `frameWidth`, `frameHeight`
- `HAND_NOT_DETECTED`: 짧은 누락은 0 프레임, 250ms 이상이면 초기화
- `RESET_SEQUENCE`
- 모든 정상 `LANDMARK_FRAME`은 같은 `frameId`의 `PREDICTION`을 반환
- warm-up 및 stride 사이에는 `symbol: "none"` 반환

테스트:

```bash
python -m unittest discover -s tests -t . -v
```

## Nginx

브라우저의 `wss://<service-host>/word` 요청을 localhost 서비스로 전달해야
한다.

```nginx
location /word {
    proxy_pass http://127.0.0.1:8767;
    proxy_http_version 1.1;
    proxy_set_header Upgrade $http_upgrade;
    proxy_set_header Connection "upgrade";
    proxy_set_header Host $host;
    proxy_read_timeout 3600s;
    proxy_send_timeout 3600s;
}
```

Nginx 변경 후 `nginx -t`와 실제 WSS 연결을 확인한다.
