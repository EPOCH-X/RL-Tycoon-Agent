# RL Handoff - 2026-03-17

이 문서는 2026-03-17 기준 추가 작업 내용을 정리한 기록이다. `RL_HANDOFF_2026-03-16.md` 이후 변경만 중심으로 적었고, 현재 기준 모델과 다음 액션도 함께 정리한다.

## 1. 현재 결론

- 여전히 기준 모델은 `models/exp21_strict_upgrade_30day_ops/final_model.zip` 이다.
- `exp22`, `exp23`은 `exp21`의 30일 장기 운영 성능을 넘지 못했다.
- responsive penalty를 강하게 넣는 방향은 성능/안정성을 해쳤고, 약하게 줄여도 `exp21`보다 낫지 않았다.
- 다음 단계는 보상 추가 튜닝보다:
  - 업그레이드 구매 순서 분석
  - trait 후보/선택 분석
  - 고득점 런의 선택 패턴 추출
  가 우선이다.

## 2. 오늘 실험 결과

### exp22_strict_upgrade_30day_responsive

- 파일: `config/experiments/exp22_strict_upgrade_30day_responsive.json`
- 목적:
  - 30일 strict-upgrade 구조 유지
  - 실제로 계측되기 시작한 `idle_penalty`, `blocked_move`, `time_penalty`를 학습에 반영
  - 화면상 멍때림/헛이동 감소 기대
- 결과:
  - 최고 mean reward: `4296.45` at `11.52M`
  - 마지막 mean reward: `2514.76`
  - `exp21` 최고/마지막 `6070.12 / 5207.32` 보다 분명히 열세
- 해석:
  - responsive penalty가 과했다.
  - 장기 운영 정책이 흔들렸고 후반 안정성이 나빠졌다.
  - 채택 비권장.

### exp23_strict_upgrade_30day_stable_responsive

- 파일: `config/experiments/exp23_strict_upgrade_30day_stable_responsive.json`
- 목적:
  - `exp21` 기준으로 회귀
  - `exp22` penalty를 더 약하게만 반영
- 결과:
  - 최고 mean reward: `5879.40` at `10.96M`
  - 마지막 mean reward: `320.84`
  - `exp22` 최고치보다는 회복됐지만 `exp21`에는 못 미침
- 해석:
  - 강한 penalty보다 약한 penalty가 낫다는 방향성은 확인
  - 그러나 후반 붕괴가 다시 커졌고 `final_model` 신뢰도가 낮음
  - 방향 검증용으로는 의미가 있지만 채택 실험으로는 실패에 가까움

## 3. 실행/학습 간극 확인 결과

오늘 다시 확인한 결론:

- 현재 남아 있는 문제는 큰 train/runtime mismatch보다는 reward/event 설계 쪽이다.
- 특히 예전에는 reward config에만 있고 실제 게임 이벤트로는 나오지 않던 항목이 있었다.

반영한 수정:

- `core/shop.py`
  - 모든 step에서 `time_penalty` 이벤트 발생
  - 막힌 이동에서 `blocked_move` 이벤트 발생
  - `ACTION_NONE`인데 일감이 남아 있으면 `idle_penalty` 이벤트 발생

의미:

- 게임 규칙을 바꾼 것은 아니고 RL 학습용 계측 신호를 정상화한 것이다.
- 이 변경은 `exp22`, `exp23`부터 실제 학습에 반영되었다.

## 4. 실시간 학습 로그

요청에 맞춰 다음 실험부터 실시간 확인 가능하도록 학습 로그를 남기게 정리했다.

- `ai/train.py`
  - `tensorboard`가 설치되어 있으면 `tb_logs/`를 자동 생성
- `exp22`, `exp23` 산출물에 실제로 `tb_logs/`가 생성됨

사용 예:

```bash
tensorboard --logdir models/exp23_strict_upgrade_30day_stable_responsive/tb_logs
```

## 5. 음식 스프라이트 문제와 수정

문제:

- `assets/sprites/food/<menu_id>.png` 형식으로 파일을 넣어도 음식이 안 보였음
- 이유는 `AssetManager`는 파일을 읽더라도, 음식을 실제로 그리는 렌더링 연결 코드가 없었기 때문

확인한 구조:

- `AssetManager`는 `assets/sprites/<entity>/<state>.png` 를 자동 탐색한다
- 하지만 현재 `food`는 엔티티 스프라이트가 아니라 텍스트 라벨로만 표시되고 있었음

반영한 수정:

- `rendering/asset_manager.py`
  - 엔티티명/상태명을 소문자로 정규화
  - `Lobster.png` 같은 대소문자 차이를 흡수
- `rendering/renderer.py`
  - 주방 `cooking` / `ready` 슬롯에 음식 아이콘 표시
  - 플레이어 운반물에 음식 아이콘 표시
  - 직원 운반물에 음식 아이콘 표시

현재 규칙:

- 파일 위치: `assets/sprites/food/<menu_id>.png`
- 예:
  - `assets/sprites/food/pasta.png`
  - `assets/sprites/food/sushi_set.png`
  - `assets/sprites/food/lobster.png`

주의:

- `menu.json`의 `id`와 파일명이 맞아야 한다.
- 음식은 이제 자동 탐색 + 실제 렌더링 둘 다 연결된 상태다.

## 6. exp24 분석용 계측 추가

새 실험:

- 파일: `config/experiments/exp24_upgrade_trait_logging_30day.json`
- 목적:
  - `exp21` 기준 30일 strict-upgrade 구조를 유지
  - 성능 개선보다 “무엇을 언제 샀는지 / 어떤 trait를 골랐는지” 로그 수집

코드 변경:

- `core/shop.py`
  - 업그레이드 구매 로그 추가
  - trait offer 로그 추가
  - trait pick 로그 추가
  - 에피소드 종료 요약에 위 로그 포함
- `ai/gym_env.py`
  - 종료 시 `episode_summary`를 `info`로 전달
  - `analysis_log_dir`가 있으면 JSONL로 저장
- `ai/train.py`
  - 학습 저장 경로 아래 `analysis_logs/`를 env에 전달

산출물:

- 학습 시 `models/exp24_upgrade_trait_logging_30day/analysis_logs/`
- env별 파일:
  - `episode_analysis_env0.jsonl`
  - `episode_analysis_env1.jsonl`
  - ...

로그에 들어가는 것:

- 최종 결과
  - `money`
  - `net_profit`
  - `customers_served`
  - `customers_lost`
  - `shop_rating`
  - `final_score`
  - `won`
- 최종 상태
  - `upgrade_levels`
  - `traits`
- 업그레이드 구매 이력
  - `day`
  - `time_elapsed`
  - `money_before/after`
  - `queue_len`
  - `upgrade_id`
  - `new_level`
  - `cost`
- trait 후보/선택 이력
  - 제시된 trait 목록
  - 각 후보 점수
  - 실제 고른 trait

## 7. 업그레이드/trait 해석 관련 현재 상태

현재 코드 기준으로 분리해서 보면:

- 업그레이드:
  - `exp19+`부터 factorized action으로 직접 학습 중
  - 따라서 “무엇을 살지”는 학습 대상
- trait:
  - 아직 RL이 직접 고르지 않음
  - `Shop.auto_select_trait()`의 휴리스틱으로 고른다

trait 휴리스틱 우선순위:

- `carry_capacity`
- `cook_time_reduction`
- `speed_bonus`
- `patience_bonus`

관련 위치:

- `core/shop.py` 의 `_score_trait_choice()`

즉 지금 당장 가능한 최선은:

1. `exp24`로 선택 로그를 수집
2. 고득점/저득점 런 차이를 분석
3. 그다음에 trait 자체를 RL 액션으로 분리할지 판단

## 8. 현재 추천 모델 / 실행 커맨드

기준 모델:

```bash
python main.py --mode ai --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

관전:

```bash
python main.py --mode watch --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

exp24 학습:

```bash
python -m ai.train --config config/experiments/exp24_upgrade_trait_logging_30day.json --save-path models/exp24_upgrade_trait_logging_30day
```

TensorBoard:

```bash
tensorboard --logdir models/exp24_upgrade_trait_logging_30day/tb_logs
```

## 9. 다음 액션

우선순위는 아래 순서가 맞다.

1. `exp24` 학습 실행
2. `analysis_logs/*.jsonl` 수집
3. 고득점 런 기준으로:
   - 업그레이드 구매 순서
   - `hire_waiter` 타이밍
   - `kitchen_expand` / `hire_chef` 진입 시점
   - trait 조합
   를 정리
4. 그 결과를 바탕으로 다음 단계 결정
   - `exp25_upgrade_order_bias`
   - `exp25_trait_policy_factorized`
   - 또는 trait 휴리스틱 개편

## 10. 요약

- 오늘 결론은 `exp21`이 여전히 기준이라는 점이다.
- `exp22`, `exp23`은 responsive penalty 방향 검증에는 의미가 있었지만 기준 모델을 대체하지 못했다.
- 음식 스프라이트는 이제 실제로 렌더링된다.
- 다음 중요한 실험은 성능 경쟁보다 선택 로그를 모으는 `exp24`다.
- 앞으로의 핵심 질문은 “무엇이 좋은지 더 학습시키는가” 이전에, “고득점 런에서 실제로 무엇을 언제 선택했는가”를 먼저 분리해내는 것이다.
