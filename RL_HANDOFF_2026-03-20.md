# RL Handoff - 2026-03-20

## 1. 오늘의 핵심 결론

- `exp28`은 현재까지 가장 강한 30일 모델로 판정했다.
- 기존 기준 모델 `exp21`은 이제 `exp28`으로 교체 가능하다고 본다.
- `exp27`의 장점이던 seated-flow 강화 방향은 맞았고, `exp28`은 그 아이디어를 안정화해서 성공했다.
- `EPOCH-8-Yongwan` 브랜치에 `main`을 머지했고, 충돌은 RL 실행 구조를 우선 보존하는 쪽으로 정리했다.
- `watch`와 `ai` 모드의 정책 차이는 모델 차이가 아니라 `deterministic / stochastic` 설정 차이였다.
- `main` 환경에서는 `exp28` 모델을 그대로 재생하려면 최소한 `MaskablePPO + runtime option + action mask` 지원이 필요하다.

## 2. exp28 결과 요약

실험 파일:
- `config/experiments/exp28_seated_flow_stability_tuned.json`

산출물:
- `models/exp28_seated_flow_stability_tuned`

평가 로그 기준:
- best mean reward: `13460.54` at `11.44M`
- last mean reward: `12496.41`
- last 5 mean: `10150.32`
- last 10 mean: `11628.32`

비교:
- `exp27`: `7703.28 / -1016.44`
- `exp26`: `5866.37 / 3360.39`
- `exp21`: `6070.12 / 5207.32`

판정:
- `exp28`은 best도 높고, last도 높고, 후반 안정성도 확보했다.
- `exp27`의 “잠재력은 높지만 불안정” 문제를 실제로 해결했다.
- 현재 기준 모델은 `exp21 final`에서 `exp28 final`로 교체 가능하다고 판단했다.

추천 실행:
```bash
python main.py --mode watch --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

## 3. exp28 Top-run 해석

상위 20런 평균:
- `final_score 49510.8`
- `served 542.8`
- `lost 8.45`
- `rating 0.81707` (약 `4.09★`)
- `queue_leave_ratio 0.2573`
- `angry_leave_ratio 0.4855`
- `fast_service_ratio 0.8994`

의미:
- `exp27` 대비 `angry_leave_ratio`가 크게 감소했다.
- `rating`과 `final_score`도 좋아졌다.
- queue leave는 여전히 높지만, 현재 목표가 high score라면 seated flow 개선으로 얻는 점수 이득이 더 크다고 해석했다.

## 4. exp28이 성공한 이유

`exp28`은 새 목표를 넣은 실험이 아니라, `exp27`을 안정화한 실험이었다.

핵심 변경:
- `learning_rate = 2e-4`
- `ent_coef = 0.005`
- service-chain reward 강도를 `exp26` 쪽으로 일부 되돌림

실무 해석:
- `exp27`이 찾은 좋은 모드를 더 자주, 더 오래 재현하게 만들었다.
- 즉 “고점 자체를 더 올린다”보다 “고점을 달성/유지하는 안정성”을 높이는 방향이 맞았다.

## 5. merge 처리

상황:
- `EPOCH-8-Yongwan` 브랜치에서 `main`을 머지
- 여러 RL 핵심 파일에서 충돌 발생

충돌 파일:
- `ai/agent.py`
- `ai/gym_env.py`
- `ai/reward.py`
- `config/settings.py`
- `core/shop.py`
- `modes/model_runtime.py`
- `modes/tournament_mode.py`
- `modes/versus_mode.py`
- `modes/watch_mode.py`

정리 방향:
- RL 실험 구조를 보존하는 쪽을 우선 채택
- 즉 아래 전제가 깨지지 않게 머지 마무리
  - `MaskablePPO`
  - runtime option 복원
  - action mask
  - strict-upgrade

머지 커밋:
- `a244149` `Merge branch 'main' into EPOCH-8-Yongwan`

검증:
- 충돌 표식 제거 완료
- 관련 파일 `py_compile` 통과
- `exp21`, `exp28` 모델 로드 + observation + action mask 1-step smoke test 통과

## 6. exp21 / exp28 실행 가능 여부

현재 `EPOCH-8-Yongwan` 브랜치에서는:
- `exp21`
- `exp28`

둘 다 실행 가능 상태로 확인했다.

확인 결과:
- `obs_len = 146`
- `mask_len = 12`
- `target_money = 1500`
- `day_limit = 30`
- `disable_auto_buy_action = True`

즉 strict-upgrade 전제와 런타임 옵션이 현재 브랜치에서 정상 복원된다.

## 7. watch / ai 모드 차이

질문:
- 왜 같은 `exp28`인데 `watch`와 `ai` 행동 패턴이 다르냐

원인:
- `watch`는 기본적으로 `deterministic=False`
- `ai`는 기본적으로 `deterministic=True`

즉 차이는 모델 차이가 아니라 추론 방식 차이였다.

정리:
- `확정적(deterministic)`:
  - 가장 확률 높은 행동만 선택
  - 비교/디버깅/제출용 확인에 적합
- `확률적(stochastic)`:
  - 정책 분포에서 샘플링
  - 다양한 패턴은 보이지만 들쭉날쭉해질 수 있음

## 8. AI 모드에 stochastic 옵션 추가

변경 파일:
- `modes/ai_mode.py`
- `main.py`

추가 내용:
- `--stochastic` 옵션 추가
- `ai mode`도 확률적으로 실행 가능

사용:
```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4 --stochastic
```

기본:
```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

## 9. watch 배경 문제 수정

문제:
- 관전 모드에서 배경이 다르게 보임

원인:
- `watch_mode.py`만 `Renderer(..., background_key="sample3")`를 사용
- 반면 `human`은 `sample1`을 사용

수정:
- `modes/watch_mode.py`의 `background_key`를 `sample1`로 변경

영향:
- 이제 `watch`가 `human`과 동일한 기본 배경을 사용

## 10. main 환경에서 exp28 실행 에러 원인

질문:
- `main` 브랜치에서 `watch`로 `exp28 final` 실행 시 왜 에러가 나는가

실제 원인:
- `MaskablePPO` 모델을 `PPO.load()`로 잘못 읽는 경로가 발생
- 에러:
  - `TypeError: MaskableActorCriticPolicy.__init__() got an unexpected keyword argument 'use_sde'`

해석:
- 이는 모델 자체 문제가 아니라
- `main` 쪽 `ai/agent.py`가 `MaskablePPO` 로드를 제대로 처리하지 못해서 난 오류다

즉 `main`에서 exp28 재생을 위해 꼭 필요한 핵심:
1. `MaskablePPO` 로드 지원
2. `action_mask` 전달
3. `train_config_used.json` 기반 runtime option 복원

## 11. main에서 exp28을 다시 뽑고 싶을 때

질문:
- `main` 환경에서 `exp28`과 같거나 더 좋은 모델을 뽑고 싶다면?

정리:
- 그냥 PPO만 다시 학습하면 `exp28` 재현이 아니다
- `exp28`은 아래 전체 구조 위에서 만들어진 모델이다
  - `MaskablePPO`
  - strict-upgrade
  - 30일/1500 기준
  - 현재 observation 구조
  - current `shop.py` / `settings.py`

최소 포팅 대상:
- `config/experiments/exp28_seated_flow_stability_tuned.json`
- `ai/train.py`
- `ai/gym_env.py`
- `ai/reward.py`
- `core/shop.py`
- `config/settings.py`

실행/관전까지 포함하면 추가 권장:
- `ai/agent.py`
- `modes/model_runtime.py`
- `modes/watch_mode.py`
- `modes/ai_mode.py`
- `modes/versus_mode.py`

핵심 결론:
- `main`에서 `exp28` 이상을 원하면, “PPO 학습”이 아니라
  “MaskablePPO + strict-upgrade + 최신 env/shop/settings 포팅”이 먼저다

## 12. main 포팅 문서

추가/업데이트한 문서:
- `EXP21_MAIN_PORTING_GUIDE_2026-03-20.md`

이 문서에 정리한 내용:
- `ai/agent.py`
- `modes/model_runtime.py`
- `modes/watch_mode.py`

가 `origin/main`과 어떻게 다른지
- 왜 `exp21` 실행에 필요했는지
- 이 수정이 `exp26/27/28`에도 거의 공통으로 적용되는지
- 어디까지가 최소 실행 조건이고, 어디부터 `gym_env/shop/settings`까지 포팅해야 하는지

## 13. 오늘 기준 추천 모델

기준 모델:
- `models/exp28_seated_flow_stability_tuned/final_model.zip`

이전 기준 모델:
- `models/exp21_strict_upgrade_30day_ops/final_model.zip`

현재 판단:
- `exp28 final`이 새 기준 모델
- `exp21 final`은 안정적 비교군으로 유지할 가치는 있음

## 14. 유용한 실행 커맨드

### exp28 관전
```bash
python main.py --mode watch --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

### exp28 AI 모드 확정적
```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```

### exp28 AI 모드 확률적
```bash
python main.py --mode ai --model models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4 --stochastic
```

### exp21 관전
```bash
python main.py --mode watch --model models/exp21_strict_upgrade_30day_ops/final_model.zip --speed 4
```

### exp21 vs exp28 비교용 토너먼트
```bash
python main.py --mode tournament --participants models/exp21_strict_upgrade_30day_ops/final_model.zip models/exp28_seated_flow_stability_tuned/final_model.zip --speed 4
```
