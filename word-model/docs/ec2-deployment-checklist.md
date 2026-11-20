# 단어 AI EC2 배포 체크리스트

## 배포 전 필수 준비

### 1. GitLab

- `feature/AI-words-JSW-server` 브랜치 pipeline 확인
- `word-ai-test`, `word-ai-build` 성공 확인
- feature 브랜치의 `word-ai-deploy`는 수동 job
- 운영 배포 전 MR 검토 및 `ai` 브랜치 병합 여부 결정

### 2. 모델 파일

체크포인트는 Git 저장소의 아래 경로에서 버전을 관리한다.

```text
word-model/models/word-19-v1/best_model.pt
```

`word-ai-test`가 저장소 파일의 체크섬과 실제 모델 로딩을 검증하고,
`word-ai-deploy`가 EC2의 다음 경로에 자동 설치한다. Docker 이미지에는
체크포인트를 중복 포함하지 않는다.

```text
/opt/sudal/ai/models/word-19-v1/best_model.pt
```

예상 SHA256:

```text
898169cf56bfbe39a3ce3b13165a8d6a53c5b873033b278ad0ab1ca7f17a72f4
```

CI 배포 실패 시 수동 복구:

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

컨테이너 UID `10003`이 bind mount의 루트 디렉터리를 통과할 수 있도록
`models`, `word-19-v1` 디렉터리는 `755`, 체크포인트 파일은 `644`인지
확인한다.

### 3. `/opt/sudal/ai/.env`

아래 항목을 확인하거나 추가한다.

```dotenv
AI_MODEL_DIR=/opt/sudal/ai/models
HANDPRACTICE_WORD_CHECKPOINT_SHA256=898169cf56bfbe39a3ce3b13165a8d6a53c5b873033b278ad0ab1ca7f17a72f4
HANDPRACTICE_WORD_TORCH_THREADS=2
HANDPRACTICE_WORD_INFERENCE_WORKERS=1
```

초기 운영은 worker 1개, Torch thread 2개를 권장한다. EC2 동시접속 부하 측정 후 조정한다.

### 4. Docker/Compose

필요 이미지:

```text
sudal-word-ai:<commit-sha>
sudal-word-ai:latest
```

예상 컨테이너:

```text
name: sudal-word-ai
bind: 127.0.0.1:8767
path: /word
model mount: /opt/sudal/ai/models:/app/models:ro
```

수동 확인:

```bash
docker compose \
  --env-file /opt/sudal/ai/.env \
  -f /opt/sudal/ai/docker-compose.yml \
  config --quiet

docker compose \
  --env-file /opt/sudal/ai/.env \
  -f /opt/sudal/ai/docker-compose.yml \
  up -d --no-deps --force-recreate word-ai

docker inspect \
  --format '{{.State.Health.Status}}' \
  sudal-word-ai

docker logs --tail 100 sudal-word-ai
```

### 5. Nginx

기존 HTTPS server block에 WebSocket proxy를 추가한다.

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

확인:

```bash
sudo nginx -t
sudo systemctl reload nginx
```

`8767`은 외부 보안 그룹에 열지 않고 localhost 바인딩을 유지한다.

### 6. 서버 자체 검증

EC2 내부:

```bash
docker exec sudal-word-ai python -c \
  "import json; from websockets.sync.client import connect; ws=connect('ws://127.0.0.1:8767/word'); ws.send('{\"type\":\"GET_CAPABILITIES\"}'); print(json.loads(ws.recv())); ws.close()"
```

확인값:

- `modelVersion == "word-19-v1"`
- `sequenceLength == 60`
- `minimumFrames == 20`
- 19개 단어와 `none`

외부 브라우저 또는 테스트 클라이언트:

```text
wss://<service-host>/word
```

### 7. 통합 검증

- 첫 20프레임까지 같은 `frameId`의 `none` 응답
- 21번째 추론 가능 프레임에서 실제 단어 및 top-3 응답
- 한 손과 두 손 payload 모두 처리
- 1~3회 짧은 미검출 후 시퀀스 유지
- 250ms 이상 미검출 후 warm-up 재시작
- 잘못된 landmark count와 frame size가 `ERROR`
- 2명 이상 동시 WebSocket 연결에서 시퀀스 분리
- 브라우저 WSS 연결과 재연결 확인

### 8. 관측값

초기 배포 후 기록:

- 추론 latency p50/p95
- 컨테이너 RSS
- EC2 CPU 사용률
- 동시 연결 수
- WebSocket error 및 reconnect 횟수
- 단어별 confidence와 오인식 사례

### 9. 롤백

- 이전 `sudal-word-ai:<commit-sha>` 이미지 보존
- 문제 발생 시 word-ai 컨테이너만 이전 이미지로 재생성
- 필요하면 Nginx `/word` location만 비활성화
- 자모 `sudal-ai`와 숫자 `sudal-number-ai`는 변경하지 않음
