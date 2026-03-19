# RL-Tycoon-Agent

> Pygame 기반 2D 레스토랑 경영 시뮬레이션 + 강화학습 에이전트 실험 프로젝트.
> 서빙, 음료, 배달, 종업원, 특성 등 10가지 시스템을 갖춘 본격 타이쿤 게임입니다.
> 팀원 5명이 각자 강화학습 브랜치에서 학습을 진행하고, 최고 성능 모델을 `main`에 병합합니다.

## 현재 상태 요약

- 엔진: Python 3.12, Pygame 2.6, Gymnasium
- RL 라이브러리: Stable-Baselines3, sb3-contrib, PyTorch
- 렌더링: 이미지 기반 스프라이트 렌더링 (AssetManager + Renderer)
- 맵 크기: 16 x 10 타일
- 타일 크기: 64 x 64 px
- 기본 UI 높이: 120 px
- 기본 게임 속도: 0.2초당 1스텝
- 지원 모드: `human`, `versus`, `watch`, `tournament`
- 지원 RL 알고리즘:
  - `PPO` (Stable-Baselines3)
  - `DQN` (Stable-Baselines3)
  - `QRDQN` (sb3-contrib)
  - `A3C` (커스텀 PyTorch)
  - `SAC` (Stable-Baselines3)
  - `DiscreteSAC` (커스텀 PyTorch)
  - `Dreamer` (세계모델 기반)
  - `MARL` (다중 에이전트)
  - `ModelBased` (모델 기반)
  - `CrossPlay` (교차 학습)
- 학습된 모델:
  - `models/ppo_v4/best_model.zip` (PPO)
  - `models/crossplay_v2/best_model.zip` (CrossPlay, PPO 기반)
  - `models/discretesac/best_model.pt` (DiscreteSAC)
- 관측 공간: 116차원 연속값 (0~1 정규화)
- 행동 공간: 7개 이산 행동

---

## Quick Start

```bash
pip install -r requirements.txt
python main.py
```

5060 환경용 패키지 세트가 필요하면:

```bash
pip install -r 5060requirements.txt
```

---

## 게임 실행

```bash
# 대화형 메뉴 (모드/일수 선택)
python main.py

# CLI 직접 실행
python main.py --mode human                                      # 솔로 30일
python main.py --mode human --days 60                             # 솔로 60일
python main.py --mode versus                                      # 대결 (AI 자동 탐색)
python main.py --mode versus --model models/ppo_v4/best_model.zip # 특정 모델과 대결
python main.py --mode watch                                       # 관전 (AI 자동 탐색)
python main.py --mode watch --model models/ppo_v4/best_model.zip --speed 2
python main.py --mode tournament --speed 3                        # 토너먼트
```

### `main.py` 옵션

| 옵션             | 설명                                                        |
| ---------------- | ----------------------------------------------------------- |
| `--mode`         | `human`, `versus`, `watch`, `tournament` (미지정 시 메뉴)   |
| `--model`        | 학습된 모델 경로. `versus`, `watch`에서 사용. 미지정 시 자동 탐색 |
| `--algo`         | 알고리즘 이름. 경로 자동 탐지가 애매할 때 지정              |
| `--speed`        | 관전/토너먼트 속도 배수 (기본 1.0)                          |
| `--target-money` | 목표 금액 강제 지정                                         |
| `--day-limit`    | 일수 강제 지정                                              |
| `--days`         | `30` 또는 `60`, `--day-limit`의 축약형                      |

---

## 플레이 모드

### 솔로 모드 (Human)

- 키보드로 직접 레스토랑을 운영합니다.
- 게임 오버 시 랭킹이 기록됩니다.

### 대결 모드 (Versus)

- 플레이어 화면을 전체 크기로 렌더링하고, AI 화면은 우측 하단 PiP(Picture-in-Picture)로 표시합니다.
- `--model`을 지정하지 않으면 `models/` 디렉토리에서 학습된 모델을 자동으로 탐색합니다.
- PPO, CrossPlay(.zip), DiscreteSAC(.pt) 등 모든 학습 알고리즘의 모델을 로드할 수 있습니다.

### 관전 모드 (Watch)

- 학습된 모델 하나를 전체 렌더링으로 관전합니다.
- `--model`을 지정하지 않으면 자동으로 모델을 탐색합니다.
- 기본 정책은 확률적(`stochastic`)으로 시작합니다.
- 조작:
  - `ESC`: 종료
  - `R`: 종료 후 재시작
  - `D`: 결정적/확률적 정책 전환
  - `↑ / ↓`: 속도 조절

### 토너먼트 모드 (Tournament)

- `models/` 아래의 학습된 모델을 자동 탐색해 최대 4개까지 동시 2x2 그리드에서 경쟁합니다.
- 알고리즘별 대표 모델을 골라 순위를 매깁니다.
- 게임 종료 시 최종 결과 오버레이(순위, 금액, 평점)가 표시됩니다.
- 기본 정책은 확률적(`stochastic`)으로 시작합니다.
- 조작:
  - `ESC`: 종료
  - `R`: 종료 후 재시작
  - `D`: 결정적/확률적 정책 전환
  - `↑ / ↓`: 속도 조절

---

## 핵심 게임 로직

### 기본 운영 흐름

1. 손님이 입장하거나 대기열에 합류합니다.
2. 빈 테이블이 있으면 착석합니다.
3. 주문을 받습니다.
4. 주방에서 음식이 조리됩니다.
5. 바텐더가 있으면 음료 주문이 추가될 수 있습니다.
6. 플레이어 또는 직원이 음식/음료를 전달합니다.
7. 결제와 팁이 정산됩니다.
8. 손님 만족도가 평점과 최종 점수에 반영됩니다.

### 점수 체계

- `net_profit`: 판매 총수입 기준 누적값
- `shop_rating_stars`: 5점 만점 환산 평점
- `final_score = net_profit * (1 + shop_rating_stars / 10)`

순수익과 평점을 동시에 올려야 최종 점수가 커집니다.

### 시간과 승리 조건

- 기본 목표 금액: `1500`
- 기본 제한 일수: `30`
- `--days 60` 또는 `--day-limit 60` 사용 가능
- 하루 길이: 60초

---

## 맵과 시설

기본 맵은 [config/map_default.json](config/map_default.json)에 정의됩니다.

- 맵 크기: `16 x 10`
- 초기 테이블: 4개
- 추가 구매 가능 테이블 슬롯: 10개 (총 최대 14개)
- 주방 카운터: 3칸
- 바 카운터: 2칸
- 쓰레기통: 1개
- 입구와 플레이어 시작 위치가 고정 배치됩니다.

### 맵 레이아웃 구조

```
┌──────────────────────────────────────────────────────────────┐
│  Row 0:  벽 (Wall)                                           │
│  Row 1:  주방 영역 – 주방카운터(3) + 바카운터(2) + 쓰레기통  │
│  Row 2:  통로 (주방↔매장 이동 구간)                          │
│  Row 3:  테이블 (초기 4개, 매장 상단)                        │
│  Row 4~5: 매장 내부 바닥 (플레이어 이동 공간)                │
│  Row 6:  확장 테이블 슬롯 (구매 가능)                        │
│  Row 7~8: 매장 하단 바닥                                     │
│  Row 9:  벽 (Wall)                                           │
└──────────────────────────────────────────────────────────────┘
```

### 타일 타입

| 코드 | 의미            |
| ---- | --------------- |
| `0`  | floor           |
| `1`  | wall            |
| `2`  | table           |
| `3`  | kitchen_counter |
| `4`  | bar_counter     |
| `5`  | trash_can       |

---

## 손님 유형

손님 유형은 [config/customers.json](config/customers.json)에 정의됩니다.

| ID        | 이름        | 인내심 | 부유도 배수 | 팁 범위 | 등장 가중치 | 평점 해금 조건 |
| --------- | ----------- | ------ | ----------- | ------- | ----------- | -------------- |
| `budget`  | 학생        | 80.0   | 0.8         | 0 ~ 1   | 5           | 0.0            |
| `normal`  | 일반인      | 65.0   | 1.0         | 1 ~ 3   | 4           | 0.0            |
| `tourist` | 관광객      | 58.0   | 1.5         | 3 ~ 8   | 2           | 0.4            |
| `wealthy` | 부유한 손님 | 50.0   | 1.8         | 4 ~ 12  | 2           | 0.5            |
| `vip`     | VIP         | 40.0   | 3.0         | 8 ~ 25  | 1           | 0.7            |
| `critic`  | 평론가      | 32.0   | 2.5         | 10 ~ 35 | 1           | 0.8            |

- 평점이 높아질수록 상위 고객군이 등장합니다.
- `marketing` 업그레이드는 부유한 손님 비중에 영향을 줍니다.
- 바텐더를 고용하면 일부 주문에 음료가 함께 붙습니다.

---

## 음식과 음료

### 음식 메뉴

음식은 [config/menu.json](config/menu.json)에 정의됩니다.

| 메뉴          | 조리시간 | 가격 | 해금조건 (순이익) |
| ------------- | -------- | ---- | ----------------- |
| 커피          | 2초      | $8   | 처음부터          |
| 샌드위치      | 3초      | $12  | 처음부터          |
| 파스타        | 4초      | $18  | 처음부터          |
| 스테이크      | 5초      | $28  | $200              |
| 초밥 세트     | 6초      | $40  | $500              |
| 랍스터        | 7초      | $55  | $1,000            |
| 와규 스테이크 | 8초      | $75  | $2,000            |
| 트러플 코스   | 10초     | $100 | $3,500            |

### 음료 시스템

음료는 [config/beverages.json](config/beverages.json)에 정의됩니다.

- 음료 시스템 해금 기준 순이익: `300`
- 음료는 바텐더 고용 후부터 활성화됩니다.

| 음료       | 제조시간 | 가격 | 해금조건 (순이익) |
| ---------- | -------- | ---- | ----------------- |
| 물         | 1초      | $2   | 처음부터          |
| 주스       | 1.5초    | $5   | 처음부터          |
| 레모네이드 | 2초      | $8   | $500              |
| 칵테일     | 3초      | $15  | $1,200            |
| 와인       | 2.5초    | $20  | $2,000            |

---

## 배달 시스템

배달 설정은 [config/delivery.json](config/delivery.json)에 정의됩니다.

- 해금 기준 순이익: `800`
- 주문 간격: `10.0`초
- 배달 시간: `12.0`초
- 가격 배수: `0.85` (수수료 15%)
- 팁 범위: `1 ~ 5`

---

## 업그레이드 시스템

업그레이드는 [config/upgrades.json](config/upgrades.json)에 정의됩니다.

### 시설(Facility) 탭

| 업그레이드  | 기본 비용 | 비용 배율 | 최대 레벨 | 해금 조건   | 효과                    |
| ----------- | --------- | --------- | --------- | ----------- | ----------------------- |
| 주방 확장   | $200      | x2.0      | 3         | 순이익 $200 | 보관 +1, 요리사 한도 +1 |
| 테이블 추가 | $120      | x1.5      | 4         | 항상        | 새 테이블 설치          |
| 조리 속도   | $120      | x2.0      | 3         | 순이익 $100 | 조리 속도 +10%/레벨     |

### 인력(Staff) 탭

| 업그레이드    | 기본 비용 | 비용 배율 | 최대 레벨 | 해금 조건   | 효과                     |
| ------------- | --------- | --------- | --------- | ----------- | ------------------------ |
| 요리사 고용   | $150      | x1.8      | 5         | 항상        | 요리사 +1 (주방 칸 필요) |
| 종업원 고용   | $200      | x2.5      | 2         | 순이익 $400 | AI 종업원 +1             |
| 바텐더 고용   | $150      | x1.0      | 1         | 순이익 $300 | 음료 서비스 개시         |
| 배달기사 고용 | $180      | x1.0      | 1         | 순이익 $800 | 배달 서비스 개시         |
| 직원 속도 업  | $150      | x2.0      | 3         | 순이익 $300 | 직원 이동속도 +25%/레벨  |

### 메뉴(Menu) 탭

- 이동속도 업: $80 (x1.8배씩 증가), 최대 3레벨, 이동속도 +20%/레벨
- 마케팅: $100 (x1.6배씩 증가), 최대 3레벨, 부유한 손님 확률 +15%/레벨

### 비용 계산 공식

```
실제 비용 = base_cost x (cost_multiplier ^ 현재_레벨)
```

---

## 특성 시스템

특성은 [config/traits.json](config/traits.json)에 정의됩니다.

- 특성 제안 주기: 5일마다
- 한 번에 제시되는 선택지 수: 3개

| ID                | 이름          | 효과                | 최대 중첩 |
| ----------------- | ------------- | ------------------- | --------- |
| `gourmet`         | 고급 음식     | 모든 음식 가격 +$2  | 5         |
| `master_chef`     | 달인          | 조리 시간 -1초      | 3         |
| `charming`        | 매력적        | 팁 +30%             | 3         |
| `efficient`       | 효율적        | 이동 속도 +15%      | 3         |
| `popular`         | 인기          | 손님 방문 빈도 +20% | 3         |
| `patient_service` | 친절한 서비스 | 손님 인내심 +5초    | 3         |
| `tip_jar`         | 팁 항아리     | 기본 팁 +$3         | 3         |

---

## 조작법

### 솔로/대결 모드

| 키                | 동작                      |
| ----------------- | ------------------------- |
| `WASD` / `방향키` | 이동                      |
| `Space` / `Enter` | 상호작용                  |
| `U`               | 업그레이드 메뉴 열기/닫기 |
| `Tab`             | 업그레이드 탭 순환        |
| `1` ~ `9`         | 업그레이드 구매           |
| `1` / `2` / `3`   | 특성 선택                 |
| `R`               | 게임 종료 후 재시작       |
| `ESC`             | 메뉴 닫기 또는 종료       |

### 상호작용(Space) 상세 동작

| 현재 들고 있는 것 | 가까이 있는 대상      | 결과                        |
| ----------------- | --------------------- | --------------------------- |
| 아무것도 없음     | 주문 대기 중인 테이블 | **주문 받기** (ORDER 획득)  |
| 주문(ORDER)       | 주방 카운터           | **주방에 전달** (조리 시작) |
| 아무것도 없음     | 주방 (READY 있음)     | **음식 수거** (FOOD 획득)   |
| 음식(FOOD)        | 해당 테이블           | **서빙** (결제 발생)        |
| 아무것도 없음     | 바 (READY 있음)       | **음료 수거** (DRINK 획득)  |
| 음료(DRINK)       | 해당 테이블           | **음료 서빙**               |
| 아무것이나        | 쓰레기통              | **아이템 폐기** (손 비우기) |

---

## 강화학습 학습

통합 런처는 [algorithms/train_launcher.py](algorithms/train_launcher.py)입니다.

### 기본 예시

```bash
python -m algorithms.train_launcher --algo PPO
python -m algorithms.train_launcher --algo PPO --days 60
python -m algorithms.train_launcher --algo QRDQN
python -m algorithms.train_launcher --algo DiscreteSAC --days 60
python -m algorithms.train_launcher --algo CrossPlay --timesteps 200000
```

### 이어서 학습

```bash
python -m algorithms.train_launcher --algo PPO --resume
```

### 평가

```bash
python -m algorithms.train_launcher --algo PPO --evaluate --model models/ppo_v4/best_model.zip
```

### 벤치마크

```bash
python -m algorithms.train_launcher --benchmark --timesteps 100000
python -m algorithms.compare_results
```

### 학습 CLI 옵션 (train_launcher)

| 옵션          | 기본값     | 설명                                        |
| ------------- | ---------- | ------------------------------------------- |
| `--algo`      | `PPO`      | 알고리즘 선택 (PPO, DQN, A3C, SAC, MARL 등) |
| `--days`      | _(config)_ | 게임 일수 (30 또는 60)                      |
| `--timesteps` | _(config)_ | 총 학습 스텝 오버라이드                     |
| `--save-path` | 자동 생성  | 모델 저장 경로                              |
| `--n-envs`    | _(config)_ | 병렬 환경 수                                |
| `--seed`      | _(config)_ | 랜덤 시드                                   |
| `--benchmark` |            | 모든 알고리즘 벤치마크 실행                 |
| `--evaluate`  |            | 학습된 모델 평가 모드                       |

---

## 관측 공간 (Observation Space) — 116차원 연속값

| 구간      | 차원 수 | 내용                                                               |
| --------- | ------- | ------------------------------------------------------------------ |
| 0~3       | 4       | 플레이어: X좌표, Y좌표, 방향, 운반 상태                            |
| 4~5       | 2       | 운반 상세: 테이블ID, 메뉴ID                                        |
| 6~9       | 4       | 이동 가능성: 상하좌우 충돌 감지                                    |
| 10~93     | 84      | 테이블 x14: [X좌표, Y좌표, 점유여부, 손님상태, 메뉴ID, 인내심비율] |
| 94~96     | 3       | 주방: 조리율, 보관율, 총 부하                                      |
| 97~102    | 6       | 랜드마크 좌표: 주방(x,y), 바(x,y), 쓰레기통(x,y)                  |
| 103~104   | 2       | 타겟 방향 벡터 (dx, dy)                                            |
| 105~107   | 3       | 대기열: 대기율, 첫 손님 인내심, 대기열 만석                        |
| 108~115   | 8       | 게임 상태: 돈, 일수, 남은시간, 평점, 업그레이드가능, 순이익, 종업원수, 바텐더 |

모든 값은 -1.0~1.0으로 정규화됩니다.

## 행동 공간 (Action Space) — Discrete(7)

| 액션        | ID  | 설명                             |
| ----------- | --- | -------------------------------- |
| UP          | 0   | 위로 이동                        |
| DOWN        | 1   | 아래로 이동                      |
| LEFT        | 2   | 왼쪽 이동                        |
| RIGHT       | 3   | 오른쪽 이동                      |
| INTERACT    | 4   | 상호작용 (테이블/주방/바)        |
| NONE        | 5   | 대기                             |
| BUY_UPGRADE | 6   | 가장 저렴한 업그레이드 자동 구매 |

---

## 보상 함수 (Reward)

Shop.step()은 이벤트 리스트를 반환하고, `ai/reward.py`의 `RewardCalculator`가
각 알고리즘 config의 `reward_shaping` 가중치를 적용합니다.

| 이벤트              | 기본 가중치 | 비고                            |
| ------------------- | ----------- | ------------------------------- |
| 주문 접수           | +8.0        | take_order                      |
| 주방 전달           | +5.0        | submit_kitchen (x전달 수)       |
| 음식 수거           | +5.0        | pickup_food                     |
| 음료 수거           | +3.0        | pickup_drink                    |
| 음식 서빙           | +15.0       | serve_food                      |
| 음료 서빙           | +8.0        | serve_drink                     |
| 손님 결제           | +1.0 x 금액 | customer_payment                |
| 잘못된 테이블       | -2.0        | wrong_table                     |
| 손님 이탈           | -15.0       | lost_customer                   |
| 업그레이드 구매     | +2.0        | buy_upgrade                     |
| 음식 해금           | +0.3 x 가격 | food_unlock                     |
| 쓰레기통 사용       | -1.0        | trash                           |
| 고아 음식 폐기      | +0.5        | trash_orphan                    |
| 대기 시작           | -0.3        | customer_waiting                |
| 대기 손님 착석      | +3.0        | waiting_customer_seated         |
| 대기 손님 이탈      | -8.0        | waiting_customer_left           |
| 벽 충돌             | -0.1        | blocked_move                    |
| 대기(아무것도 안함) | -0.3        | idle_penalty                    |
| 시간 압박           | -0.02       | time_penalty (매 스텝)          |
| 목표 달성           | +200.0      | win                             |

---

## 에이전트 로딩 시스템

`ai/agent.py`의 `load_agent()` 팩토리 함수가 모든 모드에서 사용됩니다.

- `.zip` 파일 → SB3 `PPO.load()` (TrainedAgent)
- `.pt` 파일 → 알고리즘 레지스트리 기반 `AlgorithmAgent`
- 경로에서 알고리즘 이름을 자동 탐지 (`discrete_sac`, `a3c`, `cross_play` 등)
- `--model` 미지정 시 `versus`, `watch`, `tournament` 모드 모두 `models/` 디렉토리에서 자동 탐색

---

## 알고리즘 메모

- `PPO`: 가장 안정적인 기준선. SB3 기반.
- `DQN`: 단순 baseline. SB3 기반.
- `QRDQN`: 이산 행동 공간용 분포적 DQN. sb3-contrib.
- `DiscreteSAC`: 커스텀 이산 SAC. PyTorch 직접 구현.
- `Dreamer`: 세계모델 기반 실험용.
- `CrossPlay`: 여러 알고리즘 모델을 상대 풀로 삼아 추가 학습. 최종 모델은 PPO 기반.
- `ModelBased`: 세계모델(World Model) 기반 계획 에이전트.
- `MARL`: 다중 에이전트 자기 대전 학습.

작은 이산 행동 공간과 보상 shaping 비중이 큰 환경이라, PPO/QRDQN 계열이 가장 잘 맞습니다.

---

## 프로젝트 구조

```text
RL-Tycoon-Agent/
├── main.py                     # 게임 진입점 (CLI + 대화형 메뉴)
├── requirements.txt            # pygame, gymnasium, numpy, torch, stable-baselines3
│
├── config/                     # 데이터 & 설정 (JSON + Python 상수)
│   ├── settings.py             #   전역 상수 (TILE_SIZE, FPS, 색상, 액션 코드 등)
│   ├── menu.json               #   메뉴 정의 (8종)
│   ├── customers.json          #   손님 유형 (6종)
│   ├── upgrades.json           #   업그레이드 정의 (10종)
│   ├── beverages.json          #   음료 정의 (5종)
│   ├── traits.json             #   특성 정의 (8종)
│   ├── delivery.json           #   배달 설정
│   ├── map_default.json        #   맵 레이아웃 (16x10 그리드)
│   └── train_config.json       #   RL 학습 설정 (레거시)
│
├── core/                       # 게임 코어 로직 (렌더링 없음, 순수 Python)
│   ├── entity.py               #   Entity 베이스 – 픽셀 좌표 + 스프라이트 추상화
│   ├── player.py               #   Player – 이동, 리스트 기반 운반
│   ├── customer.py             #   Customer – 상태 머신 (6단계)
│   ├── station.py              #   Table, Kitchen, BarStation – 주문/조리/음료 큐
│   ├── employee.py             #   Employee – AI NPC 종업원
│   ├── ranking.py              #   RankingManager – 로컬 랭킹 관리
│   └── shop.py                 #   Shop – 레스토랑 전체 상태 & step() (핵심 엔진)
│
├── rendering/                  # Pygame 렌더링 (코어와 완전 분리)
│   ├── asset_manager.py        #   스프라이트 시트 자동 탐색 & 프레임 분할
│   └── renderer.py             #   맵/엔티티/UI/업그레이드/특성 렌더링
│
├── modes/                      # 게임 모드 (루프 관리)
│   ├── base_mode.py            #   BaseMode – Pygame 루프 템플릿 (고정 타임스텝)
│   ├── human_mode.py           #   HumanMode – 인간 솔로 (키보드 + 연속 이동)
│   ├── versus_mode.py          #   VersusMode – 인간 vs AI (PiP 방식)
│   ├── watch_mode.py           #   WatchMode – 학습 에이전트 관전
│   └── tournament_mode.py      #   TournamentMode – 다중 AI 토너먼트 (2x2 그리드)
│
├── ai/                         # 강화학습 (팀원 수정 가능 영역)
│   ├── gym_env.py              #   TycoonEnv – Gymnasium 래퍼 (obs 116차원, act 7개)
│   ├── agent.py                #   RandomAgent / TrainedAgent / AlgorithmAgent 인터페이스
│   ├── reward.py               #   RewardCalculator – 이벤트→보상 변환
│   └── train.py                #   SB3 PPO 학습 (레거시)
│
├── algorithms/                 # RL 트레이너 구현
│   ├── base.py                 #   BaseTrainer 추상 클래스
│   ├── common.py               #   공통 유틸 (환경 생성, 설정 로드 등)
│   ├── registry.py             #   알고리즘 레지스트리
│   ├── train_launcher.py       #   통합 CLI 학습 런처
│   ├── compare_results.py      #   벤치마크 결과 비교
│   ├── ppo/                    #   PPO 트레이너 + config
│   ├── dqn/                    #   DQN 트레이너 + config
│   ├── qrdqn/                  #   QRDQN 트레이너 + config
│   ├── a3c/                    #   A3C 트레이너 + config
│   ├── sac/                    #   SAC 트레이너 + config
│   ├── discrete_sac/           #   DiscreteSAC 트레이너 + config
│   ├── dreamer/                #   Dreamer 트레이너 + config
│   ├── marl/                   #   MARL 트레이너 + config
│   ├── model_based/            #   ModelBased 트레이너 + config
│   └── cross_play/             #   CrossPlay 트레이너 + config
│
├── models/                     # 학습된 모델 저장 위치
│   ├── ppo_v4/                 #   PPO best_model.zip
│   ├── crossplay_v2/           #   CrossPlay best_model.zip (PPO 기반)
│   └── discretesac/            #   DiscreteSAC best_model.pt
│
├── assets/sprites/             # 스프라이트 시트 (자동 탐색)
├── data/                       # 랭킹 데이터
├── docs/                       # 문서
│
├── README.md                   # 이 파일
├── DEV_README.md               # 개발 메모
├── ALGORITHM_GUIDE.md          # 알고리즘별 설명과 비교
└── SPRITE_PROMPT_GUIDE.md      # 스프라이트 제작 가이드
```

---

## 각 알고리즘 설정 구조

각 알고리즘의 학습 설정은 `algorithms/<algo>/config.json` (30일)과 `config_60.json` (60일)에 분리됩니다.

```
algorithms/
├── ppo/config.json, config_60.json
├── dqn/config.json, config_60.json
├── qrdqn/config.json, config_60.json
├── a3c/config.json, config_60.json
├── sac/config.json, config_60.json
├── discrete_sac/config.json, config_60.json
├── marl/config.json, config_60.json
├── model_based/config.json, config_60.json
├── dreamer/config.json
└── cross_play/config.json
```

`--days 60` 옵션을 사용하면 자동으로 `config_60.json`이 선택됩니다.

### config.json 주요 필드 예시 (PPO)

```json
{
  "algorithm": "PPO",
  "policy": "MlpPolicy",
  "training": {
    "total_timesteps": 200000,
    "n_envs": 4,
    "seed": 42,
    "eval_freq": 5000
  },
  "hyperparameters": {
    "learning_rate": 3e-4,
    "n_steps": 2048,
    "batch_size": 64,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5
  },
  "network": {
    "net_arch": [128, 128],
    "activation_fn": "tanh"
  },
  "reward_shaping": { ... },
  "game_overrides": {
    "target_money": null,
    "day_limit": null
  }
}
```

---

## 기술 상세 (Technical Notes)

### 게임 루프 아키텍처

```
60 FPS 렌더링 루프
  ├── tick(dt)        ← 매 프레임: 연속 이동 (WASD/화살표)
  └── update()        ← 0.2초마다: 게임 로직 step (상호작용, 손님, 조리 등)
```

### 핵심 상수 (settings.py)

| 상수                       | 값   | 설명                       |
| -------------------------- | ---- | -------------------------- |
| `TILE_SIZE`                | 64   | 타일 크기 (px)             |
| `FPS`                      | 60   | 렌더링 프레임레이트        |
| `STEP_INTERVAL`            | 0.2  | 로직 갱신 주기 (초)        |
| `DAY_LENGTH`               | 60.0 | 1일 = 60초                 |
| `CUSTOMER_SPAWN_INTERVAL`  | 8.0  | 손님 배치 간격 (초)        |
| `MAX_CUSTOMERS`            | 4    | 동시 최대 착석 (초기)      |
| `KITCHEN_CAPACITY`         | 1    | 초기 요리사 수             |
| `MAX_WAITING_QUEUE`        | 4    | 대기열 최대 인원           |
| `WAITING_PATIENCE`         | 30.0 | 대기 손님 인내심 (초)      |
| `PLAYER_SPEED`             | 180  | 플레이어 이동속도 (px/s)   |
| `PLAYER_RADIUS`            | 18   | 충돌 반경 (px)             |
| `INTERACT_RANGE`           | 80   | 상호작용 거리 (px)         |
| `EMPLOYEE_SPEED`           | 120  | 종업원 이동속도 (px/s)     |
| `EMPLOYEE_ACTION_DELAY`    | 0.8  | 종업원 동작 딜레이 (초)    |
| `CUSTOMER_WALK_SPEED`      | 80   | 손님 이동속도 (px/s)       |
| `SATISFACTION_HISTORY_LEN` | 20   | 만족도 이동창 크기         |
| `LOST_CUSTOMER_PENALTY`    | 10   | 손님 이탈 벌금 ($)         |
| `DEFAULT_TARGET_MONEY`     | 1500 | 기본 목표 금액 ($)         |
| `DEFAULT_DAY_LIMIT`        | 30   | 기본 일수 제한             |

### 대결 모드 구조

- 플레이어 전체 화면 + AI PiP(우측 하단 25% 축소)
- 독립 Shop 인스턴스 2개 (각각 별도 게임 상태)
- 시간 동기화: 동일한 time_elapsed 공유
- AI는 매 스텝마다 관측→예측→행동

### 토너먼트 모드 구조

- 2x2 그리드: 최대 4개 AI 에이전트 동시 경쟁
- 축소 렌더링: 각 참가자 1/4 크기
- 게임 종료 시 최종 결과 오버레이 표시 (순위, 금액, 평점)
- `models/` 디렉토리에서 모든 학습 모델 자동 탐색

---

## 만족도 / 평점 / 점수 체계

### 만족도 (Satisfaction)

각 손님의 만족도는 서빙 완료 시 `patience_ratio`를 기준으로 계산:

- patience_ratio > 0.6: 빠른 서빙 → 만족도 0.8~1.0
- patience_ratio 0.3~0.6: 보통 서빙 → 만족도 0.5~0.8
- patience_ratio < 0.3: 느린 서빙 → 만족도 0.5 이하
- 손님 이탈: 만족도 -1.0 기록 + 벌금 $10

### 매장 평점 (Shop Rating)

- 최근 20건의 만족도를 이동 평균으로 계산 (0.0 ~ 1.0)
- 평점이 높을수록 더 부유한 손님 유형이 등장
- 평점이 낮으면 학생/일반인 손님만 옴 → 수익 저하

### 승리 조건

- **목표 달성**: 보유 금액 >= 목표 금액 ($1,500 기본) → 즉시 승리
- **시간 초과**: 30일 (실시간 1800초) 경과 시 게임 종료 → 패배
- **Versus 모드**: 먼저 목표 달성한 쪽이 승리, 시간 초과 시 금액 비교

---

## 랭킹 시스템

- 게임 종료 시 자동으로 `data/rankings.json`에 기록
- `day_limit` 기준으로 `money` 내림차순 정렬
- 게임 오버 화면에서 현재 랭킹 순위 표시

---

## 렌더링 시스템

### AssetManager

- `assets/sprites/` 디렉토리를 자동 탐색하여 스프라이트 시트를 로드합니다.
- 스프라이트가 없는 엔티티는 색상 도형으로 폴백합니다.
- 파일명이 상태명과 매칭됩니다 (예: `idle.png`, `move_up.png`).

### Renderer

- 맵 타일, 엔티티, UI 패널, 업그레이드 메뉴, 특성 팝업을 렌더링합니다.
- 배경 이미지 지원 (`sample1`, `sample2`, `sample3` 등).
- 음식/음료 아이콘을 캐릭터 머리 위에 표시합니다.
- 요리사는 주방 뒤쪽에 위치하며, 바텐더는 바 카운터에 걸친 128x64 이미지로 표시됩니다.

---

## 팀 협업 & 브랜치 전략

```
main (기초 설계 로직)
  ├── rl/팀원A   ← config 수정 + 학습 코드 + 모델 학습
  ├── rl/팀원B
  └── ...
→ 학습 결과 비교 → 최고 성능 브랜치를 main에 병합
```

### 팀원이 수정해도 되는 파일

| 파일                               | 수정 내용                     |
| ---------------------------------- | ----------------------------- |
| `algorithms/<algo>/config*.json`   | 하이퍼파라미터, 보상 가중치   |
| `ai/reward.py`                     | 보상 계산 로직                |
| `ai/train.py`                      | 학습 로직                     |
| `ai/gym_env.py`                    | 관측 공간 / 액션 공간 변경    |
| `ai/agent.py`                      | 에이전트 로딩/추론 로직       |

### 팀원이 수정하면 안 되는 파일

| 파일                            | 이유                   |
| ------------------------------- | ---------------------- |
| `core/*.py`                     | 게임 엔진 (모두 공유)  |
| `config/menu.json` 등 게임 JSON | 게임 밸런스 공유       |
| `rendering/*.py`                | 렌더링 (공유)          |
| `modes/*.py`                    | 모드 로직 (공유)       |
| `main.py`                       | 진입점 (공유)          |

---

## 관련 문서

- [ALGORITHM_GUIDE.md](ALGORITHM_GUIDE.md): 알고리즘별 설명과 비교
- [DEV_README.md](DEV_README.md): 개발 메모
- [SPRITE_PROMPT_GUIDE.md](SPRITE_PROMPT_GUIDE.md): 스프라이트 제작 가이드

---

## 의존성 (requirements.txt)

```
pygame>=2.5.0
gymnasium>=0.29.0
numpy>=1.24.0
torch>=2.0.0
stable-baselines3>=2.1.0
```

---

## 라이선스

팀 프로젝트 내부 사용.
