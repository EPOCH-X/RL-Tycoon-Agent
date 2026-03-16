# RL Post-Merge Notes - 2026-03-15

이 문서는 `main` 머지 이후 들어온 게임 로직 변경이 RL 실험 해석에 주는 영향을 정리한 메모다.

## 결론

- `exp11`, `exp14`의 기존 결과는 그대로 이어서 비교하면 안 된다.
- 이유는 단순 UI 변경이 아니라 게임 분포와 경제 밸런스가 바뀌었기 때문이다.
- 따라서 다음 순서는 `exp14` 직행이 아니라 `post-merge rebaseline -> late-game 재진입`이 맞다.

## main에서 들어온 핵심 변경

### 1. 가격 버그 수정
- `hire_waiter` 업그레이드가 지수 증가에서 `cost_list` 기반으로 바뀌었다.
- 영향:
  - 직원 고용 타이밍이 이전보다 빨라질 수 있다.
  - auto-buy heuristic의 실제 선택 분포가 달라진다.
  - late-game 운영/자동화 병목이 이전 실험과 달라질 수 있다.

### 2. 음료 주문 확률 변경
- 바텐더가 있을 때 음료가 항상 붙던 구조에서 `40%` 확률로 바뀌었다.
- 영향:
  - bartender 가치와 관련 reward 분포가 감소한다.
  - `pickup_drink`, `serve_drink` 이벤트 빈도가 낮아진다.
  - 기존 exp11/exp14 reward mix와 체감 난이도가 달라진다.

### 3. 평점/최종점수 체계 추가 또는 강화
- `shop_rating_stars`, `final_score`, `game_end` 축이 추가되었다.
- 영향:
  - 보상 설계가 순수 money/win 중심에서 score/rating 영향을 받을 수 있다.
  - main 계열 config를 그대로 쓰면 과거 실험과 reward family가 달라진다.

### 4. waiting queue 환경 추가
- `waiting_queue`, `MAX_WAITING_QUEUE`, `WAITING_PATIENCE`가 들어왔다.
- 관련 이벤트:
  - `customer_waiting`
  - `waiting_customer_seated`
  - `waiting_customer_left`
- 영향:
  - 테이블 부족 시 손님 손실 구조가 바뀐다.
  - `buy_table` 가치가 기존보다 커질 수 있다.
  - early-game/late-game 분포가 모두 변한다.

### 5. stale carry 분리 처리
- trash 시 `trash_orphan`, `stale_carry_cleared` 같은 세분화 이벤트가 생겼다.
- 영향:
  - 기존 `trash` 단일 이벤트와 의미가 달라졌다.
  - controller와 reward split을 더 명확히 할 수 있다.

## 실험 해석상 의미

- 기존 `exp11` 최고 성능은 "구 환경 + 구 경제/분포" 기준 최고다.
- 지금은 observation family는 비슷하지만 MDP가 일부 바뀌었다.
- 따라서 `exp14_lategame_curriculum`을 바로 돌려도, 성능 변화가 late-game 개선 때문인지 환경 변경 때문인지 분리할 수 없다.

## 권장 다음 순서

### Step 1. post-merge 기준선 재측정
- 새 설정: `config/experiments/exp15_postmerge_rebaseline.json`
- 목적:
  - exp11과 최대한 비슷한 reward/curriculum을 유지하되
  - main 머지 후 환경에서 early-game 기준선을 다시 얻는다.
- 원칙:
  - main에서 추가된 reward 이벤트는 비교 목적상 대부분 `0`으로 중립화
  - 따라서 성능 차이는 주로 환경/경제 변경에서 온다.

실행:
```bash
python -m ai.train --config config/experiments/exp15_postmerge_rebaseline.json --save-path models/exp15_postmerge_rebaseline
```

### Step 2. exp11 대비 차이 확인
- 비교 지표:
  - 평균 money
  - 평균 served
  - 평균 lost
  - win rate
  - upgrade 선택 시점
  - waiter/bartender 구매 빈도

### Step 3. late-game 재진입
- `exp15`가 안정적이면 그 다음에 `exp14`를 post-merge 기준으로 다시 돌린다.
- 이때는 필요하면 `exp16_postmerge_lategame.json`으로 분리하는 편이 낫다.

## 현재 판단

- 지금 가장 중요한 것은 `exp14` 결과 확인이 아니라 `post-merge baseline` 확보다.
- 그렇지 않으면 환경 변경과 late-game curriculum 효과가 섞여서 해석이 불가능해진다.
