# RL Handoff - 2026-03-18

이 문서는 2026-03-18 기준 작업 내용을 정리한 기록이다. `RL_HANDOFF_2026-03-17.md` 이후 변경분만 다루며, 현재 기준 모델과 다음 실험 방향도 함께 정리한다.

## 1. 현재 결론

- 여전히 기준 모델은 `models/exp21_strict_upgrade_30day_ops/final_model.zip` 이다.
- `exp24`는 성능 실험으로는 `exp21`을 넘지 못했지만, 업그레이드 순서와 trait 선택 패턴 분석에는 매우 유의미했다.
- `exp25`는 `exp24`에서 얻은 업그레이드 패턴을 reward로 직접 bias한 실험이었고, 결과적으로 실패했다.
- 따라서 다음 방향은:
  - `exp21` 구조로 회귀
  - `exp24` 분석을 바탕으로 업데이트한 trait heuristic 적용
  - 평점 저하 원인 진단 로그 추가
  가 맞다.

## 2. exp24 결과 분석

### 실험 개요

- 파일: `config/experiments/exp24_upgrade_trait_logging_30day.json`
- 목적:
  - `exp21` 기반 30일 strict-upgrade 구조 유지
  - 업그레이드 구매 순서 / trait 후보 / trait 선택 / 최종 결과 로그 수집

### 성능

- 산출물 위치: `models/exp24_upgrade_trait_logging_30day`
- eval 기준:
  - best mean reward: `4671.76` at `11.2M`
  - last mean reward: `3248.69`
- 해석:
  - `exp21`보다 낮다
  - 따라서 채택 모델은 아님
  - 그러나 “좋은 런이 어떤 선택을 하는가”를 보기엔 충분한 로그가 모였다

### 로그 분석 결과

- 전체 로그 에피소드: `2104`
- 상위 20런 평균:
  - `final_score 44825.53`
  - `served 538.1`
  - `lost 16.35`
  - `rating 3.625`

### 고득점 런 업그레이드 패턴

- 첫 구매는 거의 항상 `buy_table`
  - 상위 20런 중 19회
- 상위 런의 공통 구조:
  - `buy_table`
  - `hire_chef`
  - `hire_waiter`
  - `kitchen_expand`
  순으로 초반 운영 체제를 세우는 경향
- 최종적으로는 거의 항상 아래를 끝까지 올림:
  - `buy_table`
  - `hire_waiter`
  - `hire_chef`
  - `kitchen_expand`
- `hire_bartender`는 후순위이자 필수는 아님

### 고득점 런 trait 패턴

- 상위 20런 최종 스택 합계 기준:
  - `master_chef`: 45
  - `efficient`: 37
  - `patient_service`: 29
  - `gourmet`: 24
  - `charming`: 5
- 사실상 우선순위는:
  - `master_chef`
  - `efficient`
  - `patient_service`
  - `gourmet`

## 3. Trait 휴리스틱 업데이트

`exp24` 결과를 바탕으로 `core/shop.py` 의 trait 자동선택 휴리스틱을 조정했다.

### 반영 내용

- 우선순위 상향:
  - `cook_time_reduction`
  - `speed_bonus`
  - `patience_bonus`
- 보조 우선:
  - `food_price_bonus`
- 우선순위 하향:
  - `spawn_rate`
  - `tip_bonus`
  - `base_tip`
  - `carry_capacity`

### 추가 조건

- 후반일수록:
  - `master_chef`
  - `efficient`
  - `gourmet`
  가점
- 손님 이탈/대기열이 심하면:
  - `patient_service`
  가점
- 팁 계열은 초반에는 더 약하게 선택되도록 조정

의미:

- trait 선택은 아직 RL이 아니라 heuristic 기반이다.
- 따라서 지금 단계에서는 reward를 바꾸는 것보다 heuristic을 로그 기반으로 고치는 편이 안전하다.

## 4. exp25 결과와 실패 원인

### 실험 개요

- 파일: `config/experiments/exp25_upgrade_order_bias_30day.json`
- 목적:
  - `exp24`에서 추출한 고득점 업그레이드 순서를 학습이 더 잘 따르도록 약한 bias reward 추가

### 성능

- 산출물 위치: `models/exp25_upgrade_order_bias_30day`
- eval 기준:
  - best mean reward: `5496.33`
  - last mean reward: `-3309.82`
- 비교:
  - `exp21` best / last: `6070.12 / 5207.32`

### 해석

- 최고점도 `exp21`을 못 넘겼고
- 마지막은 크게 붕괴했다
- 일부 런은 `served 0 / lost 145 / final_score 0` 수준으로 완전히 망가졌다

### 왜 실패했는가

- 업그레이드 학습이 없어서가 아니라
- 업그레이드 선택 위에 직접적인 “이 순서가 좋다” bias reward를 얹은 것이 과했다
- 그 결과:
  - 정책 안정성이 무너졌고
  - 일부 런에서 업그레이드 신호를 운영보다 더 따라가며 망가지는 모드가 생겼다

### 결론

- `exp25`는 채택 비권장
- 교훈:
  - `exp24`의 분석 결과는 유효했지만
  - 그 결과를 reward에 바로 꽂는 건 좋지 않았다

## 5. TensorBoard 비교 실행

오늘 `exp21`, `exp24`, `exp25`를 한 화면에서 비교할 수 있게 TensorBoard 실행 방식을 정리했다.

비교용 명령:

```bash
tensorboard --logdir_spec exp21:models/exp21_strict_upgrade_30day_ops/tb_logs,exp24:models/exp24_upgrade_trait_logging_30day/tb_logs,exp25:models/exp25_upgrade_order_bias_30day/tb_logs --host 127.0.0.1 --port 6007
```

보면 좋은 항목:

- `eval/mean_reward`
- `rollout/ep_reward`
- `rollout/served`
- `rollout/lost`

## 6. 평점 축 분석과 exp26 방향 전환

질문:

- 최종 스코어에 평점이 중요해졌는데, 왜 평점이 너무 낮게 나오는가?

확인한 구조:

- 최종 스코어:
  - `net_profit * (1 + shop_rating_stars / 10)`
- 평점은 최근 만족도 평균
- 만족도는:
  - 빠른 서빙도 최대 `1.0`
  - 느린 서빙은 `0.2 ~ 0.38`
  - 화난 손님 이탈은 `-1.0`
- 초기 히스토리 20개가 `0.12`로 채워져 있어 초반 평점이 쉽게 오르지 않는다

관찰:

- `exp24` 전체 평균 별점은 약 `2.52★`
- 상위 20런 평균도 약 `3.63★`
- 즉 “잘한 런”도 평점이 생각만큼 높지 않다

결론:

- `exp26`은 단순히 `exp21 + trait heuristic`만이 아니라
- 평점이 낮은 원인이
  - 대기열 이탈인지
  - 테이블 angry leave인지
  - 느린 서빙 비율인지
  를 같이 볼 수 있게 설계해야 한다

## 7. exp26 설계

### 실험 파일

- `config/experiments/exp26_exp21_with_trait_rating_diagnostics.json`

### 목적

- `exp21`의 30일 strict-upgrade 장기 운영 구조 유지
- `exp24` 기반 trait heuristic 업데이트 반영
- `exp25`의 upgrade-order bias reward 제거
- 평점 저하 원인 분석용 diagnostics 추가

### reward 구조

- `exp21`과 동일
- 즉 이번 실험의 핵심은 reward 변경이 아니라:
  - trait heuristic 개선 효과 확인
  - rating root-cause diagnostics 확보

### 추가한 평점 진단 요약

`core/shop.py` 종료 summary에 아래를 추가했다:

- `waiting_customers_left`
- `waiting_customers_seated`
- `angry_table_leaves`
- `avg_served_satisfaction`
- `fast_service_count`
- `slow_service_count`
- `fast_service_ratio`
- `slow_service_ratio`
- `queue_leave_ratio`
- `angry_leave_ratio`

의미:

- `exp26`이 끝나면 단순히 평점 숫자만 보는 것이 아니라,
- 평점이 낮은 이유를 분해해서 볼 수 있다

## 8. 룰 컨트롤러 문제와 수정

문제:

- `strict-upgrade` 모델에 `--rule-controller`를 켜면 심하게 망가졌다

원인:

- `exp19+` 계열은 학습 시 `disable_auto_buy_action=true`
- 즉 `ACTION_BUY_UPGRADE(6)`을 쓰지 않는다
- 그런데 기존 `rule-controller`는 업그레이드 필요 시 강제로 `ACTION_BUY_UPGRADE`를 넣고 있었다
- 결과적으로 학습 때 없던 액션이 실전에 강제로 들어가 정책이 깨졌다

수정:

- `ai/controller.py`
  - strict 모델일 때는 `ACTION_BUY_UPGRADE`를 쓰지 않고
  - `best_choice`를 읽어 개별 업그레이드 액션으로 매핑하도록 변경

추가 판단:

- 그래도 현재 실험/평가 단계에서는 `rule-controller` 없이 보는 것이 맞다
- 이유:
  - 정책 해석이 더 쉬움
  - RL 자체가 무엇을 배웠는지 보기에 적합

## 9. 작은 수정

- `core/shop.py` 에서 `SATISFACTION_FAST_THRESHOLD` import 누락으로 `NameError`가 났고 즉시 수정했다.

## 10. 현재 추천 모델과 실행 명령

기준 모델:

```bash
python main.py --mode ai --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

관전:

```bash
python main.py --mode watch --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

대전:

```bash
python main.py --mode versus --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

exp26 학습:

```bash
python -m ai.train --config config/experiments/exp26_exp21_with_trait_rating_diagnostics.json --save-path models/exp26_exp21_with_trait_rating_diagnostics
```

## 11. 다음 액션

우선순위:

1. `exp26` 실행
2. 결과 수치 확인
3. `analysis_logs` / `episode_summary` 기준으로 아래 확인
   - 평점이 낮은 주원인이 `queue_leave`인지
   - `angry_table_leave`인지
   - `slow_service_ratio`인지
4. 그 다음 단계 결정
   - 평점이 구조적으로 너무 짜면 게임 로직/밸런스 수정 검토
   - 아니라면 RL 관측/보상 조정으로 해결

## 12. 요약

- 오늘 가장 중요한 결론은 `exp25`가 실패했고, `exp21` 구조로 돌아가야 한다는 점이다.
- 하지만 단순 회귀가 아니라:
  - `exp24` 기반 trait heuristic 개선
  - 평점 저하 원인 diagnostics
  를 붙인 `exp26`이 다음 정답에 가깝다.
- `rule-controller`는 strict 모델에 대해 현재 기본 사용 대상이 아니다.
