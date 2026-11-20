# 프론트 전달용 — 단어 수어 AI 연동 명세

- 문서 버전: `1.1.0`
- 서버 모델: `word-19-v1`
- 방식: UTF-8 JSON WebSocket
- 운영 경로: `wss://<service-host>/word`
- 사용자 범위: 카메라 앞 단일 사용자

## 프론트 구현 요청

1. MediaPipe가 검출한 raw 양손을 `left/right` 슬롯으로 전달한다.
2. `video.videoWidth`, `video.videoHeight`를 함께 전송한다.
3. 기존 latest-only WebSocket 전송과 재연결 로직을 재사용한다.
4. 단어 전용 `WordPredictionDecoder`에서 prediction을 투표해 `SIGN_CONFIRMED`를 만든다.
5. 기존 자모·숫자용 단일 손 타입과 decoder는 변경하지 않는다.

프론트는 영상, 이미지, 정답 단어를 AI 서버에 전송하지 않는다. 시퀀스 누적, 좌표 복원,
보간, 위치·속도·가속도 생성, smoothing 및 PyTorch 추론은 서버가 담당한다.

## 환경 변수

```dotenv
VITE_WORD_AI_WEBSOCKET_URL=wss://<service-host>/word
```

fallback:

```ts
const protocol = location.protocol === "https:" ? "wss:" : "ws:";
const wordAiUrl = `${protocol}//${location.host}/word`;
```

## 프론트 내부 타입

```ts
interface WordDetectedHand {
  readonly handedness: "LEFT" | "RIGHT";
  readonly score?: number;
  readonly landmarks: readonly HandLandmark[];
}

interface WordLandmarkFrame {
  readonly frameId: number;
  readonly capturedAt: number;
  readonly frameWidth: number;
  readonly frameHeight: number;
  readonly hands: {
    readonly left: WordDetectedHand | null;
    readonly right: WordDetectedHand | null;
  };
  readonly activeHandSessionId?: string;
}
```

좌표 규칙:

- `TrackedHand.landmarks` 또는 기존 `rawLandmarks`를 사용
- 화면 표시용 `light`, `stableLandmarks`는 전송하지 않음
- MediaPipe `x`, `y`, `z`를 변경하지 않음
- `frameWidth/Height`는 canvas가 아닌 `video.videoWidth/video.videoHeight`
- CSS mirror를 적용해도 `x`를 `1-x`로 바꾸지 않음
- 각 손은 정확히 21개 landmark
- 한 손만 보이면 반대 슬롯은 `null`
- 두 손이 모두 없으면 `LANDMARK_FRAME` 대신 `HAND_NOT_DETECTED`

## 요청

연결 직후:

```json
{"type":"GET_CAPABILITIES"}
```

한 손 이상 검출:

```json
{
  "type": "LANDMARK_FRAME",
  "frameId": 1234,
  "capturedAt": 1785384000123,
  "frameWidth": 1280,
  "frameHeight": 720,
  "hands": {
    "left": {
      "handedness": "LEFT",
      "score": 0.97,
      "landmarks": [
        {"x": 0.4201, "y": 0.6123, "z": -0.031}
      ]
    },
    "right": null
  }
}
```

예시 배열은 줄였지만 실제 `landmarks`는 21개다.

양손 미검출:

```json
{
  "type": "HAND_NOT_DETECTED",
  "capturedAt": 1785384000200
}
```

기존 80ms throttle을 유지한다.

문제·라운드·재시도·카메라·사용자 세션 변경:

```json
{"type":"RESET_SEQUENCE"}
```

`RESET_SEQUENCE` 전송 시 프론트 pending 요청과 단어 decoder도 함께 초기화한다.

## 응답

연결 규격:

```json
{
  "type": "CAPABILITIES",
  "modelVersion": "word-19-v1",
  "supportedSymbols": [
    "airplane", "bicycle", "moon", "motorcycle", "rain", "run",
    "ship", "star", "subway", "swim", "train", "walk", "wind", "bad",
    "bus", "car", "good", "helicopter", "sun", "none"
  ],
  "sequenceLength": 60,
  "confirmationAuthority": "FRONTEND_TEMPORAL_DECODER",
  "competitiveSymbols": [],
  "confidenceThresholds": {
    "airplane": 0.55,
    "bicycle": 0.55,
    "moon": 0.55,
    "motorcycle": 0.55,
    "rain": 0.55,
    "run": 0.55,
    "ship": 0.55,
    "star": 0.55,
    "subway": 0.55,
    "swim": 0.55,
    "train": 0.55,
    "walk": 0.55,
    "wind": 0.55,
    "bad": 0.55,
    "bus": 0.55,
    "car": 0.55,
    "good": 0.55,
    "helicopter": 0.55,
    "sun": 0.55
  },
  "inputMode": "TWO_HAND_LANDMARK_SEQUENCE",
  "minimumFrames": 20,
  "windowFrames": 60,
  "recommendedFps": 18
}
```

실제 추론:

```json
{
  "type": "PREDICTION",
  "frameId": 1234,
  "symbol": "airplane",
  "confidence": 0.83,
  "isStable": false,
  "predictedAt": 1785384000123,
  "topCandidates": [
    {"symbol": "airplane", "confidence": 0.83},
    {"symbol": "ship", "confidence": 0.09},
    {"symbol": "train", "confidence": 0.04}
  ]
}
```

warm-up 20프레임 미만, 실제 추론 사이 stride 프레임, 검출 비율 부족:

```json
{
  "type": "PREDICTION",
  "frameId": 1234,
  "symbol": "none",
  "confidence": 1.0,
  "isStable": false,
  "predictedAt": 1785384000123
}
```

모든 정상 `LANDMARK_FRAME`은 같은 `frameId`의 `PREDICTION`을 정확히 한 번 받는다.

프론트 처리:

- `none`도 in-flight 요청 완료로 처리
- `none`은 UI와 decoder 투표에 전달하지 않음
- `PREDICTION` 또는 `ERROR` 수신 시 in-flight를 해제하고 최신 pending frame 전송
- `ERROR` 한 번으로 카메라나 WebSocket을 종료하지 않음

## 전송 제어

- 동시에 진행 중인 AI 요청은 최대 1개
- 요청 중 새 프레임은 최신 1개만 pending
- 이전 pending frame은 drop
- 목표 전송률 18FPS
- 재연결 `500ms → 1s → 2s → 4s → 최대 8s`
- 재연결 후 capabilities 조회 및 session/decoder reset

프론트가 시퀀스를 모으거나 3프레임 stride를 계산하지 않는다.

## WordPredictionDecoder

기존 `DefaultContinuousSignDecoder`는 한 손 21개 landmark의 움직임·정지 상태를 전제로
하므로 단어 모드에서 직접 사용하지 않는다. 단어 모드에는 prediction 투표만 담당하는 작은
decoder를 추가한다.

초기 정책:

```ts
const WORD_PREDICTION_DECODER_CONFIG = {
  minimumConfidence: 0.55,
  candidateWindowSize: 4,
  minimumCandidateVotes: 2,
  noHandReleaseDurationMs: 250,
  differentSymbolReleaseVotes: 2,
};
```

동작:

```text
none이 아닌 실제 prediction
  → 최근 4개 보관
  → 동일 symbol 2회 이상 + 단어별 threshold 통과
  → SIGN_CONFIRMED

확정 잠금 상태
  → 양손 미검출 250ms
     또는 다른 symbol prediction 2회
  → HAND_RELEASED
```

`HAND_RELEASED` 전에는 같은 단어를 다시 확정하지 않는다.

## 단어 ID

| ID | 표시 |
|---|---|
| `airplane` | 비행기 |
| `bicycle` | 자전거 |
| `moon` | 달 |
| `motorcycle` | 오토바이 |
| `rain` | 비 |
| `run` | 달리다 |
| `ship` | 배 |
| `star` | 별 |
| `subway` | 지하철 |
| `swim` | 수영하다 |
| `train` | 기차 |
| `walk` | 걷다 |
| `wind` | 바람 |
| `bad` | 나쁘다 |
| `bus` | 버스 |
| `car` | 자동차 |
| `good` | 좋다 |
| `helicopter` | 헬리콥터 |
| `sun` | 해 |

AI 서버에 정답 ID를 보내지 않고 `SIGN_CONFIRMED.symbol`과 현재 문제 ID를 프론트에서
비교한다.

## 완료 조건

- [ ] 기존 자모·숫자 회귀 테스트 통과
- [ ] `/word` 연결 후 `GET_CAPABILITIES` 전송
- [ ] raw 양손과 실제 video 크기 전송
- [ ] 한 손 미검출 슬롯은 `null`
- [ ] CSS mirror 여부와 전송 좌표가 무관
- [ ] `none`과 `ERROR` 후 다음 pending frame 전송
- [ ] 문제 변경 및 재연결 시 서버와 decoder reset
- [ ] `none`은 prediction 투표에서 제외
- [ ] 실제 prediction 4개 중 같은 단어 2회로 `SIGN_CONFIRMED`
- [ ] 양손 미검출 250ms 또는 다른 단어 2회로 `HAND_RELEASED`
- [ ] `SIGN_CONFIRMED.symbol`과 문제 ID 비교
- [ ] 한 명만 카메라 영역에 들어오도록 UI 안내
