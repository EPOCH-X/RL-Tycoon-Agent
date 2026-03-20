# RL-Tycoon-Agent

Pygame 기반 레스토랑 타이쿤 게임에 강화학습 에이전트를 붙여 실험하는 프로젝트입니다. 사람이 직접 플레이할 수 있고, 학습된 모델과 대결하거나 관전할 수도 있습니다.

## 현재 기준 요약

- 실행 환경: Python 3.12, Pygame 2.6, Gymnasium
- 지원 모드: `human`, `versus`, `watch`, `tournament`
- 지원 알고리즘: `PPO`, `DQN`, `QRDQN`, `A3C`, `SAC`, `DiscreteSAC`, `Dreamer`, `MARL`, `ModelBased`, `CrossPlay`
- 기본 맵 크기: 16 x 10 타일
- 타일 크기: 64 px
- 게임 로직 속도: 0.2초당 1스텝
- 기본 목표 금액: `$1500`
- 기본 일수 제한: `30일`
- RL 행동 공간: `18개` 이산 행동
- RL 관측 공간: `150차원` 연속값 벡터

가장 큰 최근 변경점은 에이전트가 업그레이드 구매와 특성 선택까지 직접 학습하도록 바뀌었다는 점입니다. 예전처럼 하드코딩된 자동 구매나 자동 특성 선택에 의존하지 않습니다.

## 빠른 시작

```bash
pip install -r requirements.txt
python main.py
```

다른 패키지 세트가 필요하면 아래 파일도 사용할 수 있습니다.

```bash
pip install -r 5060requirements.txt
```

## 게임 실행

대화형 메뉴로 시작:

```bash
python main.py
```

직접 모드 지정:

```bash
python main.py --mode human
python main.py --mode human --days 60
python main.py --mode versus
python main.py --mode versus --model models/ppo_v5/best_model.zip
python main.py --mode watch --model models/discretesac_v2/best_model --speed 2
python main.py --mode tournament --speed 3
```

### CLI 옵션

| 옵션 | 설명 |
| --- | --- |
| `--mode` | `human`, `versus`, `watch`, `tournament` |
| `--model` | 불러올 학습 모델 경로. 주로 `versus`, `watch`에서 사용 |
| `--algo` | 알고리즘 이름 수동 지정. 경로 자동 판별이 애매할 때 사용 |
| `--speed` | `watch`, `tournament` 속도 배수 |
| `--target-money` | 목표 금액 강제 지정 |
| `--day-limit` | 일수 제한 강제 지정 |
| `--days` | `30` 또는 `60`, `--day-limit` 축약형 |

## 플레이 모드

### Human

- 플레이어가 직접 레스토랑을 운영합니다.
- 게임 오버 결과는 랭킹 시스템에 기록됩니다.

### Versus

- 사람과 AI가 각각 동일한 규칙의 독립된 매장을 운영합니다.
- AI 화면은 우측 하단 PiP로 표시됩니다.
- 모델을 지정하지 않으면 `models/` 아래의 학습 모델을 자동 탐색하거나 선택 UI를 띄웁니다.
- AI가 업그레이드나 특성을 선택하면 화면 중앙에 잠깐 알림이 뜹니다.

### Watch

- 학습된 모델 1개를 전체 화면으로 관전합니다.
- 최근 행동과 행동 분포를 우측 패널에서 볼 수 있습니다.
- `D` 키로 결정적 정책과 확률적 정책을 전환할 수 있습니다.

### Tournament

- `models/` 아래에서 최대 4개 모델을 자동 탐색해 동시에 경쟁시킵니다.
- 각 참가자 매장은 2x2 레이아웃으로 렌더링됩니다.
- 각 패널에 업그레이드와 특성 선택 알림이 표시됩니다.

## 조작법

### Human / Versus 인간 플레이어

| 키 | 동작 |
| --- | --- |
| `WASD`, 방향키 | 이동 |
| `Space`, `Enter` | 상호작용 |
| `U` | 업그레이드 메뉴 열기/닫기 |
| `Tab` | 업그레이드 탭 순환 |
| `1` ~ `9` | 업그레이드 구매 |
| `1`, `2`, `3` | 특성 선택 |
| `R` | 게임 종료 후 재시작 |
| `ESC` | 종료 또는 메뉴 닫기 |

### Watch / Tournament

| 키 | 동작 |
| --- | --- |
| `D` | 결정적/확률적 정책 전환 |
| `↑`, `↓` | 속도 조절 |
| `R` | 종료 후 재시작 |
| `ESC` | 종료 |

## 게임 시스템 개요

- 손님은 입장 후 빈 테이블이 있으면 착석하고, 없으면 대기열에서 기다립니다.
- 플레이어나 종업원이 주문 접수, 주방 전달, 음식 수거, 서빙을 처리합니다.
- 바텐더를 고용하면 음료 시스템이 활성화됩니다.
- 업그레이드와 특성은 매장 운영 능력과 RL 정책 모두에 큰 영향을 줍니다.
- 평점과 순이익이 최종 스코어에 반영됩니다.

현재 기본 설정 기준 주요 수치:

- 최대 동시 착석 손님: `4`
- 최대 대기열: `6`
- 특성 제안 주기: `4일마다`
- 특성 선택지 수: `3개`
- 기본 테이블 수: `4개`
- 구매 가능한 추가 테이블 수: `10개`

## RL 인터페이스

현재 환경은 [ai/gym_env.py](ai/gym_env.py) 기준으로 다음 인터페이스를 사용합니다.

- 행동 공간: `Discrete(18)`
- 관측 공간: `Box(shape=(150,))`
- 업그레이드 구매 액션: `9개`
- 특성 선택 액션: `3개`

행동 구성:

| 구분 | 액션 수 | 설명 |
| --- | --- | --- |
| 이동/기본 행동 | 6 | 상하좌우, 상호작용, 대기 |
| 업그레이드 구매 | 9 | 업그레이드 종류별 개별 액션 |
| 특성 선택 | 3 | 제시된 특성 3개 중 하나 선택 |

관측에는 다음 정보가 포함됩니다.

- 플레이어 위치, 방향, 운반 상태
- 테이블별 손님 상태와 인내심
- 주방 상태
- 주방, 바, 쓰레기통 위치
- 현재 목표 방향 벡터
- 대기열 상태
- 돈, 남은 시간, 평점, 종업원 수
- 업그레이드별 레벨, 구매 가능 여부, 해금 여부
- 현재 특성 선택 상태와 제시된 특성 정보

## 학습

통합 런처는 [algorithms/train_launcher.py](algorithms/train_launcher.py) 입니다.

### 기본 학습

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
python -m algorithms.train_launcher --algo DiscreteSAC --resume
```

### 평가

```bash
python -m algorithms.train_launcher --algo PPO --evaluate --model models/ppo_v5/best_model.zip
python -m algorithms.train_launcher --algo DiscreteSAC --evaluate --model models/discretesac_v2/best_model
```

### 벤치마크

```bash
python -m algorithms.train_launcher --benchmark --timesteps 100000
python -m algorithms.compare_results
```

## 모델 파일 규칙

프로젝트는 아래 파일들을 자동 탐색 대상으로 사용합니다.

- `best_model.zip`
- `final_model.zip`
- `best_model.pt`
- `final_model.pt`
- `train_config_used.json`

`watch`, `versus`, `tournament`, `cross-play`는 위 파일 규칙을 기준으로 모델과 알고리즘을 자동 판별합니다.

## 폴더 구조

| 경로 | 역할 |
| --- | --- |
| [main.py](main.py) | 게임 진입점, 메뉴 및 모드 선택 |
| [ai](ai) | Gym 환경, 에이전트 로더, 보상 계산 |
| [algorithms](algorithms) | 알고리즘별 트레이너, 통합 런처 |
| [core](core) | 매장, 손님, 직원, 플레이어 등 핵심 게임 로직 |
| [modes](modes) | human, versus, watch, tournament 모드 |
| [rendering](rendering) | 에셋 로딩, 화면 렌더링 |
| [config](config) | 메뉴, 손님, 업그레이드, 특성, 맵 설정 |
| [models](models) | 학습 결과물 저장 폴더 |
| [docs](docs) | 프로젝트 메모와 문서 |

## 추가 문서

- [DEV_README.md](DEV_README.md): 현재 코드 구조와 RL 인터페이스 정리
- [SPRITE_PROMPT_GUIDE.md](SPRITE_PROMPT_GUIDE.md): 스프라이트 제작 가이드
