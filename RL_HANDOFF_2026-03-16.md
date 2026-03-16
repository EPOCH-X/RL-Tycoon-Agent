# RL Handoff - 2026-03-16

이 문서는 2026-03-16 기준 RL 실험 진행 내용, 코드 변경, 현재 채택 모델, 다음 액션을 한 번에 이어받기 위한 기록이다.

## 1. 현재 결론

- 10일 이하 short-horizon 실험군보다 `30일 strict-upgrade` 계열이 현재 게임 방향과 더 잘 맞는다.
- 현재 가장 유망한 모델은 `models/exp21_strict_upgrade_30day_ops/final_model.zip` 이다.
- `exp21`은 실제 30일 운영 기준에서:
  - `buy_table`, `hire_waiter`, `kitchen_expand`, `hire_chef`를 끝까지 올리는 장기 운영 패턴을 만들었고
  - queue 손실을 크게 줄였고
  - 승리/최종점수도 가장 잘 나왔다.
- 실행 모드와 학습 모드의 환경 불일치 문제는 수정했다.

## 2. 왜 실험 축이 바뀌었는가

`main` 머지 이후 게임 자체가 바뀌었다.

- 가격/업그레이드 로직 변경
  - `hire_waiter`, `buy_table` 등 일부 업그레이드가 `cost_list` 기반으로 동작
- 대기열(`waiting_queue`) 추가
  - `customer_waiting`
  - `waiting_customer_seated`
  - `waiting_customer_left`
- 평점/최종점수 체계 강화
  - `shop_rating`
  - `shop_rating_stars`
  - `final_score`
  - `game_end`
- 손님 스폰과 장기 운영 분포가 평점과 대기열에 더 민감해짐

따라서 구환경 기준의 `exp11`, `exp14`는 그대로 비교하면 안 되고, post-merge 재기준선이 필요했다.

관련 문서:
- `RL_POSTMERGE_2026-03-15.md`
- `RL_HANDOFF_2026-03-13.md`

## 3. 오늘 진행한 실험 흐름

### exp15_postmerge_rebaseline

- 파일: `config/experiments/exp15_postmerge_rebaseline.json`
- 목적:
  - post-merge 환경에서 old exp11 계열 early-game baseline 재측정
- 특징:
  - 새 reward 축(queue/rating/final score)을 거의 0으로 중립화
  - `target_money=150`, `day_limit=8`
- 결과 해석:
  - post-merge 적응 자체는 되었지만
  - 운영은 좋아져도 `win`으로 닫는 정책은 약함

### exp16_postmerge_finish_bias

- 파일: `config/experiments/exp16_postmerge_finish_bias.json`
- 목적:
  - exp15보다 종료 현금 확보와 win 마감을 강화
- 특징:
  - `buy_upgrade` 축소
  - `win`과 `customer_payment` 소폭 강화
- 결과 해석:
  - 운영 품질은 개선
  - 하지만 승률 개선은 충분하지 않음

### exp17_postmerge_queue_cash_alignment

- 파일: `config/experiments/exp17_postmerge_queue_cash_alignment.json`
- 목적:
  - queue/rating/final score 축을 일부 복원
  - queue-aware 운영 + cash 확보 동시 학습
- 추가 코드:
  - 관측에 queue 관련 값 추가
- 결과 해석:
  - 10일 계열 중 가장 좋은 실험
  - `exp16` 대비 큰 개선
  - baseline/short-horizon 기준으로는 성공

### exp18_upgrade_choice_factorized

- 파일: `config/experiments/exp18_upgrade_choice_factorized.json`
- 목적:
  - 업그레이드 선택을 개별 액션으로 분해해 “무엇을 살지” 학습
- 추가 코드:
  - action space를 7 -> 12로 확장
  - 개별 업그레이드 액션 추가
    - `buy_table`
    - `hire_waiter`
    - `hire_bartender`
    - `kitchen_expand`
    - `hire_chef`
  - 업그레이드 비용/가능 여부를 관측에 추가
- 결과 해석:
  - 설계 의도는 맞았지만
  - 기존 `ACTION_BUY_UPGRADE`를 남겨둬서 factorized choice 학습이 반쯤 무력화
  - `exp17`보다 낫지 않았음

### exp19_upgrade_choice_strict

- 파일: `config/experiments/exp19_upgrade_choice_strict.json`
- 목적:
  - 학습에서 기존 auto-buy shortcut을 끄고 factorized upgrade 선택을 강제
- 추가 코드:
  - `disable_auto_buy_action` env option 도입
- 결과 해석:
  - `exp18`보다 명확히 개선
  - `hire_waiter`를 실제로 배우기 시작
  - 다만 10일 계열 전체 기준으로는 아직 불안정

### exp20_strict_upgrade_play_quality

- 파일: `config/experiments/exp20_strict_upgrade_play_quality.json`
- 목적:
  - strict upgrade 유지 + 화면상 덜 멍청해 보이는 즉시 반응성 회복
- 특징:
  - service loop, stale carry, idle, blocked move 쪽 보상 조정
- 결과 해석:
  - 기대만큼 좋아지지 않음
  - waiter 선택이 다시 약해지고
  - strict 계열 기준으로는 `exp19`보다 후퇴

### exp21_strict_upgrade_30day_ops

- 파일: `config/experiments/exp21_strict_upgrade_30day_ops.json`
- 목적:
  - 30일 실제 게임 기준으로 strict upgrade 구조를 유지한 장기 운영 학습
- 특징:
  - `target_money=1500`, `day_limit=30`
  - `total_timesteps=12,000,000`
  - queue/rating/final score 중심 장기 보상
- 결과 해석:
  - 현재 가장 성공적인 실험
  - `exp19`를 30일 환경에 태운 것보다 크게 우수
  - `final_model.zip`이 `best_model.zip`보다 실제 운영 지표에서 더 좋게 나옴

## 4. 오늘 들어간 핵심 코드 변경

### 4.1 업그레이드 선택 학습 구조 추가

수정 파일:
- `config/settings.py`
- `core/shop.py`
- `ai/gym_env.py`
- `modes/watch_mode.py`

변경 내용:
- action 수 확장
  - 기존 7개 -> 현재 12개
- 신규 액션 추가
  - `ACTION_BUY_TABLE`
  - `ACTION_HIRE_WAITER`
  - `ACTION_HIRE_BARTENDER`
  - `ACTION_KITCHEN_EXPAND`
  - `ACTION_HIRE_CHEF`
- `BUY_UPGRADE` 자동구매 외에 개별 선택 경로 추가
- `cost_list` 기반 업그레이드 비용 계산을 공통화
- 관측에 아래 정보 추가
  - queue ratio
  - queue full flag
  - oldest waiting patience
  - 대표 업그레이드 구매 가능 여부
  - 대표 업그레이드 다음 비용 비율

### 4.2 strict factorized upgrade용 env option 추가

수정 파일:
- `core/shop.py`

변경 내용:
- `disable_auto_buy_action` 옵션 추가
- 학습에서 `ACTION_BUY_UPGRADE`를 마스크에서 제거 가능

### 4.3 실행 모드와 학습 모드 설정 불일치 수정

수정 파일:
- `modes/model_runtime.py` 신규 추가
- `modes/ai_mode.py`
- `modes/watch_mode.py`
- `modes/versus_mode.py`

문제:
- `exp19+`는 학습 시 `disable_auto_buy_action=true`로 돌렸는데
- 실행 모드는 기본 `Shop()`으로 생성해서 이 옵션이 반영되지 않았음
- 그래서 실제 관찰 시 학습 당시와 다른 액션 집합이 열려 있었음

수정 내용:
- 각 실행 모드가 모델 폴더의 `train_config_used.json`에서
  - `game_overrides`
  - `env_options`
  를 읽어 `Shop(...)` 생성 시 함께 적용하도록 변경
- `WatchMode`도 now action mask를 `predict()`에 전달

현재 이 수정 덕분에:
- `exp19`, `exp20`, `exp21` 같은 strict-upgrade 모델이
- 실전 실행에서도 학습 당시와 같은 action availability를 유지한다.

## 5. 현재 추천 실행 모델

가장 먼저 볼 모델:

```bash
python main.py --mode ai --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

관전 모드:

```bash
python main.py --mode watch --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

비교용:

```bash
python main.py --mode ai --model models/exp17_postmerge_queue_cash_alignment/best_model.zip --speed 4
```

```bash
python main.py --mode ai --model models/exp19_upgrade_choice_strict/best_model.zip --speed 4
```

## 6. 현재 모델 선택 가이드

### short-horizon 기준

- 가장 안정적인 short-horizon 계열: `exp17`

### strict-upgrade 구조 확인용

- 구조적 전환의 시작점: `exp19`

### 실제 30일 운영 기준

- 현재 1순위: `exp21 final`

## 7. 해석상 주의점

- `best_model`이 항상 최종 채택 모델은 아니다.
- 특히 `exp21`은 샘플 재평가 기준으로 `final_model`이 더 좋았다.
- 30일 실험은 에피소드가 길어서 재평가 시간이 길고, 샘플 수도 적으면 변동성이 있다.
- 실행 화면에서 “멍청해 보인다”는 체감과 reward/score가 항상 일치하지 않는다.
- strict-upgrade 계열은 반드시 실행 시에도 `train_config_used.json`의 `env_options`가 반영되어야 한다.

## 8. 지금 바로 해야 할 일

1. `exp21 final`을 실제로 관찰
2. 아래 항목 체크
   - 초반 확장 순서가 자연스러운가
   - `hire_waiter`를 적절한 시점에 사는가
   - 대기열이 꽉 찼을 때 병목을 완화하는가
   - 중반 이후 멈춤/왕복/헛행동이 반복되는가
   - 후반 30일 동안 rating과 queue가 유지되는가
3. 실제 관찰상 여전히 멍청해 보이는 구간을 기록
   - 예: 주문 직전 테이블을 무시
   - 예: 새 테이블만 늘리고 waiter 미구매
   - 예: 주방 병목 상태에서 table/kitchen 선택 꼬임

## 9. 다음 실험 후보

### 후보 A. exp22_waiter_table_alignment_30day

조건:
- `exp21`이 overall 좋지만
- 실제 관찰에서 여전히 table/waiter 타이밍이 어색할 때

방향:
- waiter 필요 신호 추가
- table/waiter 선택 조건을 더 직접적으로 관측에 반영
- queue loss와 staff bottleneck 신호 강화

### 후보 B. exp22_playback_quality_30day

조건:
- `exp21` score는 좋지만 화면 체감이 여전히 나쁠 때

방향:
- 왕복 이동, idle, stale carry, blocked move를 더 구조적으로 억제
- 단, `exp20`처럼 short-loop 보상을 과도하게 키우지 말 것

### 후보 C. exp22_finalscore_aligned_30day

조건:
- 실제 게임 기준 최종 점수를 더 직접적으로 최적화하고 싶을 때

방향:
- `game_end`, `rating_delta`, `final_score_delta` 비중 재조정
- win보다 final score 최적화 성향 강화

## 10. 요약

- 오늘의 최종 결론은 `30일 strict-upgrade` 방향이 맞다는 것이다.
- 가장 중요한 기술적 수정은:
  - factorized upgrade action 추가
  - auto-buy strict 차단 옵션 추가
  - 실행 모드가 학습 config를 따라가도록 수정
- 현재 채택 후보는 `exp21 final`
- 다음 단계는 새 실험을 서둘러 돌리는 것보다, `exp21 final`을 실제 관찰하면서 남은 병목을 정확히 분리하는 것이다.
