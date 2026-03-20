# DEV_README

현재 코드 기준 개발 레퍼런스입니다. 예전 설계 메모 대신, 실제 동작 중인 구조와 RL 인터페이스를 빠르게 파악할 수 있도록 정리했습니다.

## 아키텍처 요약

실행 진입점:

- [main.py](main.py): CLI 파싱, 메뉴 UI, 모드 선택

핵심 게임 로직:

- [core/shop.py](core/shop.py): 게임 진행, 업그레이드, 특성, 승패, 이벤트 발생
- [core/customer.py](core/customer.py): 손님 상태 전이
- [core/employee.py](core/employee.py): 종업원 자동 행동
- [core/player.py](core/player.py): 플레이어 운반 및 상호작용

렌더링:

- [rendering/asset_manager.py](rendering/asset_manager.py): 스프라이트 로딩
- [rendering/renderer.py](rendering/renderer.py): 맵, 손님, UI, 상태 배지 렌더링

RL:

- [ai/gym_env.py](ai/gym_env.py): Gymnasium 환경, 관측 생성, 행동 공간 정의
- [ai/agent.py](ai/agent.py): 모델 로딩, 경로 기반 알고리즘 감지
- [ai/reward.py](ai/reward.py): 보상 계산
- [algorithms/train_launcher.py](algorithms/train_launcher.py): 통합 학습 런처
- [algorithms/registry.py](algorithms/registry.py): 알고리즘 레지스트리

모드:

- [modes/human_mode.py](modes/human_mode.py)
- [modes/versus_mode.py](modes/versus_mode.py)
- [modes/watch_mode.py](modes/watch_mode.py)
- [modes/tournament_mode.py](modes/tournament_mode.py)

## 현재 게임 규칙 스냅샷

설정 기준 파일은 [config/settings.py](config/settings.py), [config/map_default.json](config/map_default.json), [config/upgrades.json](config/upgrades.json), [config/traits.json](config/traits.json) 입니다.

- 기본 목표 금액: `1500`
- 기본 일수 제한: `30일`
- 하루 길이: `60초`
- 스텝 간격: `0.2초`
- 최대 동시 착석 손님: `4`
- 최대 대기열: `6`
- 기본 테이블: `4개`
- 추가 구매 가능 테이블: `10개`
- 특성 제안 주기: `4일`
- 특성 선택지 수: `3개`

## 현재 RL 인터페이스

### 행동 공간

[config/settings.py](config/settings.py) 기준 `NUM_ACTIONS = 18` 입니다.

| 액션 인덱스 | 의미 |
| --- | --- |
| `0` | 위 이동 |
| `1` | 아래 이동 |
| `2` | 왼쪽 이동 |
| `3` | 오른쪽 이동 |
| `4` | 상호작용 |
| `5` | 대기 |
| `6` | 이동속도 업 구매 |
| `7` | 주방 확장 구매 |
| `8` | 요리사 고용 |
| `9` | 조리 속도 업 |
| `10` | 테이블 추가 |
| `11` | 마케팅 구매 |
| `12` | 종업원 고용 |
| `13` | 바텐더 고용 |
| `14` | 직원 속도 업 |
| `15` | 특성 1 선택 |
| `16` | 특성 2 선택 |
| `17` | 특성 3 선택 |

핵심 변경점:

- 업그레이드 구매가 더 이상 단일 추상 액션이 아닙니다.
- 특성 선택도 더 이상 자동 선택되지 않습니다.
- 에이전트가 운영 중 구매와 특성 선택을 직접 학습합니다.

### 관측 공간

[ai/gym_env.py](ai/gym_env.py) 기준 관측 길이는 기본 맵에서 `150` 입니다.

구성은 다음과 같습니다.

| 구간 | 차원 수 | 설명 |
| --- | --- | --- |
| 플레이어 상태 | 4 | 위치, 방향, 운반 타입 |
| 운반 상세 | 2 | 테이블 ID, 메뉴 ID |
| 이동 가능성 | 4 | 상하좌우 충돌 가능 여부 |
| 테이블 상태 | 84 | 최대 14개 테이블 x 6값 |
| 주방 상태 | 3 | 조리 수, 보관 수, 부하율 |
| 주요 지점 좌표 | 6 | 주방, 바, 쓰레기통 좌표 |
| 목표 방향 | 2 | 현재 우선 목표 방향 벡터 |
| 대기열 상태 | 3 | 대기율, 첫 손님 인내심, 만석 여부 |
| 게임 상태 | 8 | 돈, 날짜, 남은 시간, 평점, 구매 가능성 등 |
| 업그레이드 상세 | 27 | 9개 업그레이드 x 3값 |
| 특성 선택 상태 | 7 | 선택 중 여부 + 3개 선택지 정보 |

합계:

```text
4 + 2 + 4 + 84 + 3 + 6 + 2 + 3 + 8 + 27 + 7 = 150
```

주의할 점:

- 관측 코드에서 업그레이드 해금 체크는 현재 `unlock_net_profit` 키를 읽고 있습니다.
- 설정 JSON은 `unlock_profit` 을 사용하므로, 문서화할 때는 실제 설정 키와 코드 사용처를 구분해서 봐야 합니다.

## 모드별 동작 포인트

### Human

- 플레이어 직접 조작
- 랭킹 저장 대상

### Versus

- 사람과 AI가 각각 독립된 `Shop` 을 가집니다.
- AI 모델 미지정 시 선택 UI를 띄웁니다.
- AI가 업그레이드나 특성을 선택하면 중앙 토스트를 띄웁니다.

### Watch

- 단일 모델 관전용
- 최근 행동, 행동 확률, 에피소드 내 행동 분포를 표시합니다.
- 18개 행동 이름을 모두 반영합니다.

### Tournament

- 최대 4개 모델 자동 로드
- 각 참가자별 독립 `Shop`
- 각 패널에 구매와 특성 토스트를 표시합니다.

## 모델 로딩 규칙

[ai/agent.py](ai/agent.py), [algorithms/cross_play/trainer.py](algorithms/cross_play/trainer.py) 기준 현재 자동 탐색 대상은 다음과 같습니다.

- `best_model.zip`
- `final_model.zip`
- `best_model.pt`
- `final_model.pt`

추가 메타데이터:

- `train_config_used.json`: 알고리즘명, 학습 설정 복원에 사용

로딩 방식:

- `.zip`: SB3 계열 우선 로딩
- `.pt`: 커스텀 트레이너 계열 로딩
- 경로 문자열에 `ppo`, `discretesac`, `crossplay` 같은 힌트가 있으면 알고리즘 자동 감지

## CrossPlay 메모

[algorithms/cross_play/trainer.py](algorithms/cross_play/trainer.py) 기준 현재 CrossPlay 는 다음 흐름으로 동작합니다.

- `models/` 아래의 완료된 모델들을 자동 수집
- 상대 풀을 만든 뒤 learner 를 하나 선택
- SB3 계열은 SelfPlayEnv 에서 직접 재학습
- 커스텀 계열은 원래 트레이너를 환경 오버라이드 방식으로 재사용

최근 반영된 내용:

- `final_model.zip` 도 자동 탐색 대상에 포함됨
- reward 설정은 실제 이벤트 이름 기준으로 정리됨

## 렌더링 관련 최근 상태

[rendering/renderer.py](rendering/renderer.py) 기준 최근 시각 변경점:

- 의자는 손님이 없어도 항상 렌더링됩니다.
- 착석 손님 위에 `주문 대기`, `음식 대기` 상태 배지가 표시됩니다.
- 메인 메뉴는 회전하며 떨어지는 요리사 스프라이트를 사용합니다.

## 문서 업데이트 원칙

이 문서는 예전 기획 메모보다 코드 기준 사실을 우선합니다. 아래 파일이 바뀌면 이 문서도 함께 확인하는 것이 좋습니다.

- [config/settings.py](config/settings.py)
- [ai/gym_env.py](ai/gym_env.py)
- [core/shop.py](core/shop.py)
- [modes/versus_mode.py](modes/versus_mode.py)
- [modes/watch_mode.py](modes/watch_mode.py)
- [modes/tournament_mode.py](modes/tournament_mode.py)
- [algorithms/cross_play/trainer.py](algorithms/cross_play/trainer.py)
