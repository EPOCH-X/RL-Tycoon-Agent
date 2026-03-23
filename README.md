# RL-Tycoon-Agent

Pygame 기반 레스토랑 타이쿤 게임에 강화학습 실험 환경을 결합한 프로젝트입니다. 플레이어가 직접 운영할 수 있는 게임을 유지하면서, 동일한 규칙 위에서 여러 강화학습 알고리즘을 학습시키고 `versus`, `watch`, `tournament` 모드로 성능을 비교할 수 있도록 구성했습니다.

이 프로젝트의 핵심은 단순 이동 제어가 아니라 운영 의사결정까지 학습한다는 점입니다. 현재 에이전트는 이동, 상호작용, 대기뿐 아니라 업그레이드 구매 9종과 특성 선택 3종까지 포함한 총 18개 행동을 직접 선택합니다.

## 프로젝트 요약

- 엔진: Python 3.12, Pygame 2.6, Gymnasium
- 기본 게임 규칙: 30일, 60일, 1스텝 `0.2초`
- 맵 크기: `16 x 10` 타일, 타일 크기 `64px`
- RL 행동 공간: `Discrete(18)`
- RL 관측 공간: `Box(shape=(150,))`
- 실행 모드: `human`, `versus`, `watch`, `tournament`
- 학습 런처: `python -m algorithms.train_launcher`
- 현재 등록 알고리즘: `PPO`, `DQN`, `Rainbow`, `QRDQN`, `A3C`, `SAC`, `MARL`, `ModelBased`, `DiscreteSAC`, `Dreamer`, `DreamerV3`, `MuZero`, `CrossPlay`

## 최종 결과 요약

현재 프로젝트 마무리 기준 토너먼트 결과에서 가장 좋은 성능을 보인 모델은 `DiscreteSAC` 계열입니다.

```text
╔══════════════════════════════════════════
║      토너먼트 최종 결과
║ 1st #1 discretesac_조영 스코어: 53,944 WIN
║ 2nd #2 crossplay_영곤   스코어: 25,697 WIN
║ 3rd #3 ppo_민승         스코어: 18,132
```

핵심 해석:

- 현재 저장된 실험 결과 기준 최고 성능 모델은 `models/discretesac_조영/best_model.pt`
- 교차 대전 기반 추가학습 모델인 `CrossPlay`도 강한 성능을 보였지만, 최종 스코어는 `DiscreteSAC`보다 낮음
- `PPO`는 안정적으로 동작하지만 이번 프로젝트의 최종 경쟁 결과에서는 상위 1위 모델을 넘지 못함

즉, 이 프로젝트의 최종 강화학습 성능 결론은 `DiscreteSAC > CrossPlay > PPO` 입니다.

## 왜 이 프로젝트가 의미 있었는가

일반적인 게임 RL 예제와 달리 이 프로젝트는 다음 특징을 갖습니다.

- 실시간 운영형 시뮬레이션에서 장기 의사결정을 학습함
- 업그레이드 구매와 특성 선택이 정책 안에 포함됨
- 최종 평가를 `final_score` 기준으로 비교할 수 있음
- 여러 알고리즘을 같은 게임 환경에서 비교 가능함
- 토너먼트 모드로 최대 4개 모델을 같은 화면에서 직접 검증 가능함

## 설치

기본 설치:

```bash
pip install -r requirements.txt
```

GPU 환경에 따라 PyTorch는 별도 설치가 필요할 수 있습니다. 현재 `requirements.txt`에는 기본 패키지만 정리되어 있고, CUDA 예시는 파일 내 주석으로 남겨져 있습니다.

## 실행 방법

메인 메뉴 실행:

```bash
python main.py
```

직접 모드 실행:

```bash
python main.py --mode human
python main.py --mode human --days 60
python main.py --mode versus
python main.py --mode watch
python main.py --mode tournament --speed 3
```

모델 지정 예시:

```bash
python main.py --mode versus --model models/ppo/best_model.zip
python main.py --mode watch --model models/dqn/best_model.zip --speed 2
python main.py --mode watch --model models/discretesac/best_model.pt --algo DiscreteSAC
```

## 게임 모드

### Human

- 플레이어가 직접 레스토랑을 운영합니다.
- 주문, 조리, 서빙, 업그레이드, 특성 선택을 모두 수동으로 수행합니다.

### Versus

- 플레이어와 AI가 각각 독립된 매장을 동시에 운영합니다.
- 학습된 모델을 지정하면 사람과 AI의 운영 성능을 바로 비교할 수 있습니다.
- AI는 확률적 정책으로 동작하도록 구성되어 있어 실제 학습된 행동 성향을 보기 좋습니다.

### Watch

- 단일 AI 모델을 관전하는 모드입니다.
- 현재 행동, 행동 분포, 속도 배수를 함께 볼 수 있습니다.

### Tournament

- `models/` 아래 학습된 모델을 자동 탐색해 최대 4개까지 동시에 경쟁시킵니다.
- 같은 조건에서 최종 스코어를 비교하기 때문에 프로젝트 최종 평가에 가장 적합한 모드입니다.
- 현재는 중앙 오버레이형 스코어보드를 사용합니다.

## 조작 키

### Human / Versus 플레이어

- `WASD`, 방향키: 이동
- `Space`, `Enter`: 상호작용
- `U`: 업그레이드 메뉴 열기/닫기
- `Tab`: 업그레이드 탭 순환
- `1` ~ `9`: 업그레이드 구매
- `1`, `2`, `3`: 특성 선택
- `R`: 게임 종료 후 재시작
- `ESC`: 종료 또는 메뉴 닫기

### Watch / Tournament

- `D`: 결정적/확률적 정책 전환
- `↑`, `↓`: 속도 조절
- `R`: 종료 후 재시작
- `ESC`: 종료

## 강화학습 환경 설계

현재 RL 환경은 `ai/gym_env.py`를 기준으로 구현되어 있습니다.

### 행동 공간

총 18개 이산 행동:

- 이동 4개: 위, 아래, 왼쪽, 오른쪽
- 기본 행동 2개: 상호작용, 대기
- 업그레이드 구매 9개: 각 업그레이드별 개별 행동
- 특성 선택 3개: 제시된 특성 3개 중 하나 선택

이 구조 덕분에 에이전트는 단순히 손님을 따라다니는 정책이 아니라, 언제 업그레이드를 사고 어떤 특성을 선택할지까지 포함한 운영 전략을 학습합니다.

### 관측 공간

관측 길이는 기본 맵 기준 `150`입니다.

포함 정보:

- 플레이어 위치, 방향, 운반 상태
- 이동 가능 여부
- 테이블별 손님 상태와 인내심
- 주방 상태
- 핵심 상호작용 지점 좌표
- 현재 목표 방향 벡터
- 대기열 상태
- 돈, 날짜, 평점 등 게임 상태
- 업그레이드별 상태
- 특성 선택 진행 상태

### 점수 체계

최종 평가는 `core/shop.py` 기준 `final_score`를 사용합니다.

현재 계산식:

$$
final\_score = net\_profit \times \left(1 + \frac{shop\_rating\_stars}{10}\right)
$$

여기서 `net_profit`은 현재 코드상 실제 순이익이라기보다 매장 총 수입 `total_earned`를 사용합니다. 따라서 높은 매출과 높은 평점을 동시에 만드는 정책이 유리합니다.

## 학습 방법

통합 학습 진입점은 `algorithms/train_launcher.py` 입니다.

### 새 학습 시작

```bash
python -m algorithms.train_launcher --algo PPO
python -m algorithms.train_launcher --algo DQN
python -m algorithms.train_launcher --algo QRDQN
python -m algorithms.train_launcher --algo Rainbow
python -m algorithms.train_launcher --algo A3C
python -m algorithms.train_launcher --algo SAC
python -m algorithms.train_launcher --algo MARL
python -m algorithms.train_launcher --algo ModelBased
python -m algorithms.train_launcher --algo DiscreteSAC
python -m algorithms.train_launcher --algo Dreamer
python -m algorithms.train_launcher --algo DreamerV3
python -m algorithms.train_launcher --algo MuZero
python -m algorithms.train_launcher --algo CrossPlay
```

60일 설정 예시:

```bash
python -m algorithms.train_launcher --algo PPO --days 60
python -m algorithms.train_launcher --algo DiscreteSAC --days 60
```

### 이어서 학습

SB3 계열 `.zip` 모델 예시:

```bash
python -m algorithms.train_launcher --algo PPO --resume --save-path models/ppo_민승
python -m algorithms.train_launcher --algo CrossPlay --resume --save-path models/crossplay_영곤
```

커스텀 `.pt` 계열은 `checkpoint.pt`가 있어야 재개가 가능합니다. `best_model.pt`만 있는 경우는 추론용 가중치로는 사용할 수 있지만 학습 재개용 상태가 없을 수 있습니다.

### 평가

```bash
python -m algorithms.train_launcher --algo PPO --evaluate --model models/ppo_민승/best_model.zip
python -m algorithms.train_launcher --algo CrossPlay --evaluate --model models/crossplay_영곤/best_model.zip
python -m algorithms.train_launcher --algo DiscreteSAC --evaluate --model models/discretesac_조영/best_model
```

## 모델 파일 형식

이 프로젝트는 두 가지 주요 모델 저장 형식을 사용합니다.

### `.zip`

- 주로 Stable-Baselines3 계열 모델에서 사용
- 예: `PPO`, `DQN`, `CrossPlay`
- 대표 파일명: `best_model.zip`, `final_model.zip`, `checkpoint.zip`

### `.pt`

- 커스텀 PyTorch 트레이너 계열에서 사용
- 예: `DiscreteSAC`, `A3C`, `SAC`, `ModelBased`, `Dreamer`
- 대표 파일명: `best_model.pt`, `final_model.pt`, `checkpoint.pt`

주의:

- `.zip`는 보통 SB3 계열로 로드됩니다.
- `.pt`는 커스텀 트레이너로 로드됩니다.
- 경로만으로 알고리즘 판별이 애매하면 `--algo`를 함께 지정하는 것이 안전합니다.

## 현재 모델 폴더 상태

현재 `models/` 폴더에는 다음 실험 결과가 있습니다.

- `crossplay_영곤`
- `discretesac_조영`
- `discrete_sac`
- `ppo_민승`
- `ppo_정은`

현재 확인된 대표 파일:

- `models/discretesac_조영/best_model.pt`
- `models/crossplay_영곤/best_model.zip`
- `models/crossplay_영곤/checkpoint.zip`
- `models/ppo_민승/best_model.zip`
- `models/ppo_민승/checkpoint.zip`

즉, 프로젝트 최종 보고 기준으로는 다음처럼 정리할 수 있습니다.

- 최고 성능 모델: `discretesac_조영`
- 재현성과 학습 로그 보존 측면에서 관리가 잘 된 모델: `crossplay_영곤`, `ppo_민승`

## 폴더 구조

```text
RL-Tycoon-Agent/
├─ main.py
├─ requirements.txt
├─ ai/
│  ├─ agent.py
│  ├─ gym_env.py
│  ├─ reward.py
│  └─ train.py
├─ algorithms/
│  ├─ registry.py
│  ├─ train_launcher.py
│  ├─ common.py
│  ├─ compare_results.py
│  ├─ ppo/
│  ├─ dqn/
│  ├─ rainbow/
│  ├─ qrdqn/
│  ├─ a3c/
│  ├─ sac/
│  ├─ discrete_sac/
│  ├─ marl/
│  ├─ model_based/
│  ├─ dreamer/
│  ├─ dreamerv3/
│  ├─ muzero/
│  └─ cross_play/
├─ core/
│  ├─ shop.py
│  ├─ customer.py
│  ├─ employee.py
│  ├─ player.py
│  └─ ranking.py
├─ modes/
│  ├─ human_mode.py
│  ├─ versus_mode.py
│  ├─ watch_mode.py
│  └─ tournament_mode.py
├─ rendering/
│  ├─ asset_manager.py
│  └─ renderer.py
├─ config/
│  ├─ settings.py
│  ├─ customers.json
│  ├─ delivery.json
│  ├─ menu.json
│  ├─ map_default.json
│  ├─ traits.json
│  └─ upgrades.json
├─ assets/
├─ data/
├─ models/
└─ venv/
```

### 디렉토리 역할

- `ai/`: RL 환경, 모델 로더, 보상 계산
- `algorithms/`: 알고리즘별 트레이너와 통합 학습 런처
- `core/`: 게임 규칙과 엔티티 중심의 핵심 로직
- `modes/`: 실제 플레이 및 관전 모드
- `rendering/`: 렌더링과 에셋 관리
- `config/`: 게임 규칙과 밸런스 설정 JSON
- `models/`: 학습 결과 저장 폴더
- `data/`: 랭킹 등 저장 데이터

## 프로젝트 결론

이 프로젝트는 레스토랑 운영 시뮬레이션 위에서 여러 강화학습 알고리즘을 실험하고, 동일한 환경에서 직접 비교할 수 있게 만든 RL 실험 플랫폼입니다.

최종 결과 기준 핵심 결론은 다음과 같습니다.

- 운영형 의사결정을 포함한 이산 행동 문제에서 `DiscreteSAC`가 가장 높은 성능을 보였다.
- `CrossPlay`는 경쟁형 추가학습 구조 덕분에 강한 성능을 보였고 실험 관리도 안정적이었다.
- `PPO`는 여전히 기본 비교 기준선으로 유효하지만, 최종 토너먼트 최고 성능은 아니었다.
- `tournament` 모드는 프로젝트 결과를 가장 직관적으로 검증하는 최종 평가 수단이었다.

따라서 이 프로젝트의 최종 대표 모델은 `DiscreteSAC`이며, 프로젝트 핵심 성과 역시 강화학습 기반 운영 전략 학습과 알고리즘 비교 실험에 있습니다.
