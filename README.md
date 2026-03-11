# RL-Tycoon-Agent

> 레스토랑 경영 시뮬레이션 + 강화학습 에이전트 실험 프로젝트.
> 서빙 효율 및 수익 최적화를 목표로 하는 RL 에이전트의 행동 변화 분석.

Python + **Pygame** 기반 2D 레스토랑 매니지먼트 게임으로, **사람**과 **RL 에이전트** 모두 플레이 가능합니다.

---

## Quick Start

```bash
# 1) 의존성 설치
pip install -r requirements.txt

# 2) 게임 실행 (인간 솔로)
python main.py --mode human

# 3) 인간 vs AI 대결 (랜덤 에이전트)
python main.py --mode versus

# 4) 인간 vs 학습된 AI
python main.py --mode versus --model models/best_model.zip

# 5) RL 에이전트 학습
python -m ai.train --timesteps 200000 --save-path models
```

---

## Controls

| Key               | Action                                       |
| ----------------- | -------------------------------------------- |
| Arrow Keys / WASD | 이동 (상하좌우)                              |
| Space / Enter     | 상호작용 (주문 받기 / 주방 전달 / 음식 서빙) |
| R                 | 재시작 (게임 오버 시)                        |
| ESC               | 종료                                         |

---

## How to Play

플레이어는 레스토랑의 **서버(웨이터)** 역할입니다.

1. **주문 받기** – 손님이 테이블에 앉으면(`?!` 표시) → 테이블 앞으로 이동 → Space
2. **주방에 전달** – 주문을 들고(`ORDER`) → 주방 카운터로 이동 → Space
3. **음식 수거** – 요리 완료(초록색 `Ready`) 시 → 주방 카운터 → Space
4. **서빙** – 음식을 들고(`FOOD`) → 해당 테이블로 이동 → Space
5. **수익** – 빠른 서빙 → 높은 만족도 → 부유한 손님 → 더 많은 돈!
6. **목표** – 30일(기본) 내에 목표 금액($5000) 달성!

메뉴·손님·업그레이드 등 모든 수치는 `config/` 폴더의 **JSON 파일**만 수정하면 됩니다.

---

## Game Mechanics

### 손님 흐름

```
손님 도착(테이블) → 서버가 주문 받음 → 주방에 전달 → 주방 조리 → 서버가 음식 수거 → 서빙 → 결제
```

### 만족도 시스템

- 각 손님의 **인내심**(patience)이 시간에 따라 감소
- 빠른 서빙 → 높은 만족도 → 팁 보너스
- 느린 서빙 → 손님 이탈 → 벌금 ($30)
- 전체 **매장 평점**(shop_rating)은 최근 만족도의 이동 평균

### 손님 유형 (재산 등급)

| 유형    | 재산배율 | 인내심 | 팁     | 등장 조건      |
| ------- | -------- | ------ | ------ | -------------- |
| Budget  | 0.8×     | 45초   | $0-3   | 항상           |
| Normal  | 1.0×     | 35초   | $2-8   | 항상           |
| Wealthy | 1.8×     | 28초   | $5-20  | 매장 평점 60%+ |
| VIP     | 3.0×     | 22초   | $10-50 | 매장 평점 60%+ |

### 주방 시스템

- 최대 3개 동시 조리 (업그레이드 가능)
- 메뉴별 조리 시간: Coffee(3초) ~ Sushi Set(15초)
- 조리 완료된 음식은 픽업 대기열에 보관

### 승리 조건

- **목표 달성**: 목표 금액($5000) 도달 → 즉시 승리
- **시간 초과**: 30일(1800초) 경과 → 최종 금액으로 평가

---

## Project Structure

```
RL-Tycoon-Agent/
├── main.py                     # 게임 진입점 (CLI)
├── requirements.txt
│
├── config/                     # ★ 모든 게임 데이터 (JSON + 상수)
│   ├── settings.py             #   전역 상수 (타일 크기, 색상, 액션 등)
│   ├── menu.json               #   메뉴 정의 (조리 시간, 가격)
│   ├── customers.json          #   손님 유형 (인내심, 재산 배율, 팁)
│   ├── upgrades.json           #   업그레이드 정의 (비용, 효과)
│   └── map_default.json        #   기본 맵 레이아웃 (그리드, 테이블, 주방 배치)
│
├── core/                       # ★ 게임 코어 로직 (렌더링 없음)
│   ├── entity.py               #   Entity 베이스 – 도트 ↔ 스프라이트 추상화
│   ├── player.py               #   플레이어 (이동, 주문/음식 운반)
│   ├── customer.py             #   손님 (상태 머신, 인내심, 만족도)
│   ├── station.py              #   테이블 & 주방 (주문 큐, 조리 큐)
│   └── shop.py                 #   Shop – 하나의 레스토랑 전체 상태 & step()
│
├── rendering/                  # ★ Pygame 렌더링 시스템
│   ├── asset_manager.py        #   스프라이트 시트 로더 & 애니메이션 매니저
│   └── renderer.py             #   맵·엔티티·UI 통합 렌더러
│
├── modes/                      # ★ 게임 모드
│   ├── base_mode.py            #   Pygame 게임 루프 스켈레톤
│   ├── human_mode.py           #   Mode 1: 인간 솔로
│   └── versus_mode.py          #   Mode 2: 인간 vs AI 분할 화면
│
├── ai/                         # ★ 강화학습 컴포넌트
│   ├── gym_env.py              #   Gymnasium 환경 래퍼 (TycoonEnv)
│   ├── agent.py                #   에이전트 인터페이스 (Random / Trained)
│   └── train.py                #   SB3 PPO 학습 스크립트
│
└── assets/
    └── sprites/                # Phase 3: 스프라이트 시트 드롭 위치
```

---

## Development Phases

| Phase              | 내용                                                          |
| ------------------ | ------------------------------------------------------------- |
| **Phase 1** (현재) | 그래픽 에셋 없이 기본 도형으로 코어 엔진 완성                 |
| **Phase 2**        | (RL 팀) Gymnasium + SB3 학습 / (디자인 팀) 픽셀아트 에셋 생성 |
| **Phase 3**        | 학습된 모델 + 스프라이트 에셋 최종 병합                       |

---

## Adding Sprite Assets (Phase 3)

스프라이트는 `assets/sprites/<entity_name>/<state>.png` 형식으로 배치합니다.
각 PNG는 **수평 스프라이트 시트**이며 TILE_SIZE(64×64) 기준으로 자동 분할됩니다.

```
assets/sprites/
├── player/
│   ├── idle.png
│   ├── move_up.png
│   ├── move_down.png
│   ├── move_left.png
│   └── move_right.png
├── customer/
│   ├── idle.png
│   └── angry.png
└── ...
```

`AssetManager`가 시작 시 자동 탐색하므로 **코드 수정 없이** 이미지 파일만 추가하면
해당 Entity의 렌더링이 도형 → 스프라이트로 전환됩니다.

---

## Technical Notes

- **고정 타임스텝**: 게임 로직은 `STEP_INTERVAL=0.2s` (5 steps/sec) 단위로 진행
- **렌더링**: 60 FPS로 별도 실행 (로직과 분리)
- **관측 공간**: 플레이어 위치/운반 상태 + 테이블별 상태 + 주방 상태 + 금액/일자/평점
- **RL 액션**: `Discrete(6)` – 상/하/좌/우/상호작용/대기
- **Versus 모드**: 두 개의 독립 Shop 인스턴스를 동기화된 시계로 동시 시뮬레이션

---

## Contributing (팀원용)

1. `main` 브랜치에서 개인 브랜치 생성
2. `config/` JSON만 수정하여 밸런스 실험 가능
3. `ai/train.py` 하이퍼파라미터 조정 후 학습 → `models/` 에 저장
4. 최고 성능 에이전트를 `main`에 PR로 병합
