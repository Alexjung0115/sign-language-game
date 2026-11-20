# 수어의 달인

웹캠으로 수어 동작을 실시간 채점하고, 학습과 게임, WebRTC 기반 1대1 대전을 제공하는 팀 프로젝트입니다.

## 내가 맡은 일

- **AI** · 웹캠 수어에서 잘못된 동작을 걸러내는 단어 채점 모델과 판정 로직을 개발했습니다.
- **인프라** · EC2 한 대에 Spring Boot 백엔드, MySQL, AI WebSocket 서버 3종, coturn을 분리해 배포했습니다. 서비스별 GitLab CI/CD와 Nginx의 HTTPS·WSS 진입점, 모니터링과 백업·복구 절차를 구성했습니다.
- **백엔드** · 출석, 테스트, 솔로 게임, 대전의 경험치 지급을 공통 보상 서비스로 통합했습니다. 최근 테스트에서 틀린 문제를 바탕으로 출제 가중치를 계산하는 API도 구현했습니다.

## 사용 기술

| 분야 | 기술 | 적용한 작업 |
| --- | --- | --- |
| 수어 단어 채점 | Python, OpenCV, MediaPipe, PyTorch, Transformer, ONNX Runtime | MediaPipe 랜드마크를 입력으로 단어 채점 모델을 학습하고 INT8 ONNX로 변환해 CPU 추론을 검증 |
| 백엔드 | Java, Spring Boot, Spring Data JPA, MySQL, Flyway | 보상 지급과 오답 기반 출제 API를 구현하고 게임 데이터의 저장 구조를 관리 |
| 인프라·실시간 통신 | AWS EC2, Linux, Docker Compose, GitLab CI/CD, Nginx, WebSocket, WebRTC, coturn | 서비스별 배포를 구성하고 HTTPS·WSS 접속과 대전에 필요한 TURN 서버를 운영 |
| 모니터링·부하 테스트 | Prometheus, Grafana, k6 | 서버·컨테이너 상태를 확인하고 랭킹 API 병목을 재현 |

## 랭킹 API 개선

사용자 1만 명과 게임 결과 30만 건을 넣은 격리 환경에서 k6로 랭킹 API 병목을 재현했습니다. MySQL 실행 계획을 확인하고 커버링 인덱스를 적용한 뒤, 같은 부하 조건에서 p95 응답 시간이 약 4.44초에서 185ms로 줄었습니다.

## 코드 둘러보기

| 경로 | 내용 |
| --- | --- |
| [`deploy/`](deploy) · [`coturn/`](coturn) · [`monitoring/`](monitoring) | 배포 구성, TURN과 운영 모니터링 |
| [`backend/`](backend) | Spring Boot API와 게임 데이터 처리 |
| [`ai/`](ai) · [`frontend/`](frontend) | 수어 인식과 사용자 화면 |

[팀 시연 영상](https://youtu.be/Y6J_x2KZ0X8)

SSAFY GitLab에서 가져온 코드 스냅샷입니다. 학습 데이터와 모델 가중치, 운영 설정과 내부 협업 자료를 제외했습니다. 실제 수어 추론을 실행하려면 각 모델과 실행 환경을 별도로 준비해야 합니다.
