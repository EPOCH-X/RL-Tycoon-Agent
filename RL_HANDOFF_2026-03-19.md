# RL Handoff - 2026-03-19

## 1. 오늘의 핵심 결론

- `exp27`은 잠재력은 가장 강하게 나왔지만, 후반 안정성이 크게 무너졌다.
- 현재 채택 기준 모델은 여전히 `exp21 final`.
- 다음 실험은 `exp27`의 seated-flow 방향은 유지하되, 보상 강도와 학습 강도를 낮춘 안정화 버전 `exp28`로 진행한다.
- 앞으로 TensorBoard에는 `served/lost/rating/final_score` 등 커스텀 지표도 같이 기록된다. 단, 이 변경은 **다음에 새로 시작하는 학습부터** 적용된다.

## 2. exp27 결과 요약

실험 파일:
- `config/experiments/exp27_seated_flow_stability_30day.json`

산출물:
- `models/exp27_seated_flow_stability_30day`

평가 로그 기준:
- best mean reward: `7703.28` at `11.92M`
- last mean reward: `-1016.44`
- last 5 eval 평균: `3574.53`

비교 기준:
- `exp21` best / last: `6070.12 / 5207.32`
- `exp26` best / last: `5866.37 / 3360.39`

해석:
- 최고점만 보면 지금까지 중 가장 높다.
- 하지만 후반 붕괴가 매우 커서 `final_model` 신뢰도가 낮다.
- `exp27`은 채택 실험이라기보다, “seated customer flow 강화는 high-end 성능을 올릴 수 있다”는 방향 검증 실험으로 보는 게 맞다.

권장 사용:
- `best_model.zip`만 참고
- `final_model.zip`은 비권장

실행:
```bash
python main.py --mode ai --model models/exp27_seated_flow_stability_30day/best_model.zip --speed 4
```

```bash
python main.py --mode watch --model models/exp27_seated_flow_stability_30day/best_model.zip --speed 4
```

## 3. exp27 diagnostics 해석

상위 20런 평균:
- `final_score 49106.13`
- `served 553.55`
- `lost 10.65`
- `shop_rating 0.784455` (약 `3.92★`)
- `queue_leave_ratio 0.1534`
- `angry_leave_ratio 0.6932`
- `fast_service_ratio 0.8868`

`exp26` top20과 비교:
- `final_score`, `served`, `lost`, `rating`은 거의 비슷하거나 소폭 열세
- `angry_leave_ratio`는 개선됨
- `queue_leave_ratio`는 악화됨

핵심 해석:
- `exp27`은 의도대로 앉힌 손님의 angry leave를 줄이는 데는 성공했다.
- 대신 queue 관리가 약해지면서 front side 손실이 늘었다.
- 즉 `seated flow`를 잡으려다 `queue flow`를 일부 희생한 패턴이다.

## 4. exp28 설계

새 실험 파일:
- `config/experiments/exp28_seated_flow_stability_tuned.json`

설계 의도:
- `exp27`의 seated-flow 관측은 유지
- trait heuristic / rating diagnostics 유지
- service-chain reward 강도는 `exp26` 쪽으로 약하게 되돌림
- 학습률과 entropy를 낮춰 후반 붕괴를 줄임

핵심 변경:
- `learning_rate = 2e-4`
- `ent_coef = 0.005`
- `take_order = 4.0`
- `submit_kitchen = 6.0`
- `pickup_food = 6.0`
- `serve_food = 17.5`

실행:
```bash
python -m ai.train --config config/experiments/exp28_seated_flow_stability_tuned.json --save-path models/exp28_seated_flow_stability_tuned
```

의도:
- `exp27`의 high-end 잠재력은 유지
- `queue_leave_ratio` 재악화를 줄임
- `best`와 `final` 차이를 줄임

## 5. TensorBoard 커스텀 지표 추가

수정 파일:
- `ai/train.py`

추가 내용:
- `EpisodeSummaryTensorboardCallback` 추가
- episode 종료 시 env `info["episode_summary"]`를 읽어 TensorBoard scalar로 기록

새로 기록되는 태그:
- `custom/served`
- `custom/lost`
- `custom/shop_rating`
- `custom/final_score`
- `custom/queue_leave_ratio`
- `custom/angry_leave_ratio`
- `custom/fast_service_ratio`
- `custom/slow_service_ratio`
- `custom/avg_served_satisfaction`

중요:
- 이 변경은 **이미 실행 중이던 exp27에는 적용되지 않음**
- 이유: `model.learn()` 시작 시점에 콜백이 등록되기 때문
- 따라서 `exp28`부터는 위 커스텀 지표를 TensorBoard에서 볼 수 있음

## 6. 오늘 다시 정리된 판단 기준

질문:
- “최종 목표가 high score인데, `exp21/26/27` 비교에 이 점이 반영됐나?”

답:
- 반영은 되어 있었다.
- 다만 학습 중 공통 그래프는 `eval/mean_reward` 중심이었고,
- 실험 해석 단계에서 `final_score`, `rating`, `queue/angry leave`를 함께 봤다.

현재 더 올바른 해석 기준:
1. `eval/mean_reward`
2. `top-run final_score`
3. `shop_rating`
4. `queue_leave_ratio`
5. `angry_leave_ratio`

즉 앞으로는 “reward가 좋아 보이냐”보다 “high score에 실제로 더 가까워졌냐”를 더 강하게 기준으로 잡는 것이 맞다.

## 7. 오늘 확인/정리된 구조적 사실

### 7.1 `core/shop.py`에서 누적된 핵심 변경 축

1. strict-upgrade 지원
- `disable_auto_buy_action`
- action mask에서 auto-buy 슬롯 차단

2. 실제 reward event 계측
- `time_penalty`
- `blocked_move`
- `idle_penalty`

3. 업그레이드/trait 로그
- `upgrade_purchase_log`
- `trait_offer_log`
- `trait_pick_log`

4. 평점 원인 진단
- `angry_table_leaves`
- `fast_service_count`
- `slow_service_count`

5. 종료 요약 확장
- `fast_service_ratio`
- `slow_service_ratio`
- `queue_leave_ratio`
- `angry_leave_ratio`

### 7.2 초창기 구조는 auto-buy 중심이었음

정리:
- 초기에는 RL이 “지금 살까 말까”만 결정했고,
- 실제로 무엇을 살지는 `Shop` 내부 휴리스틱이 골랐다.
- 이후 `strict-upgrade` 실험으로 바뀌면서, RL이 개별 업그레이드를 직접 선택하게 되었다.

## 8. 현재 기준 모델과 추천

현재 채택 모델:
- `models/exp21_strict_upgrade_30day_ops/final_model.zip`

이유:
- best뿐 아니라 last 안정성까지 가장 낫다.
- runtime에서도 비교적 일관성이 좋다.

연구용으로 볼 가치가 큰 모델:
- `models/exp26_exp21_with_trait_rating_diagnostics/best_model.zip`
- `models/exp27_seated_flow_stability_30day/best_model.zip`

역할:
- `exp26`: trait heuristic + rating diagnostics 기반
- `exp27`: seated flow 강화의 잠재력 확인용

## 9. 다음 액션

1. `exp28` 학습
2. TensorBoard에서 아래를 같이 보기
- `eval/mean_reward`
- `custom/served`
- `custom/lost`
- `custom/shop_rating`
- `custom/angry_leave_ratio`
- `custom/queue_leave_ratio`
3. `exp27` 대비
- 후반 붕괴가 줄었는지
- queue 손실이 회복됐는지
- angry leave 개선이 어느 정도 유지되는지 확인

## 10. 참고 커맨드

### exp21 / exp26 / exp27 비교 TensorBoard
```bash
tensorboard --logdir_spec exp21:models/exp21_strict_upgrade_30day_ops/tb_logs,exp26:models/exp26_exp21_with_trait_rating_diagnostics/tb_logs,exp27:models/exp27_seated_flow_stability_30day/tb_logs,exp28:models/exp28_seated_flow_stability_tuned/tb_logs --host 127.0.0.1 --port 6008
```

### exp26 실행
```bash
python main.py --mode ai --model models/exp26_exp21_with_trait_rating_diagnostics/best_model.zip --speed 4
```

```bash
python main.py --mode watch --model models/exp26_exp21_with_trait_rating_diagnostics/best_model.zip --speed 4
```

### exp21 vs exp26 AI 대전
```bash
python main.py --mode tournament --participants models/exp21_strict_upgrade_30day_ops/final_model.zip models/exp26_exp21_with_trait_rating_diagnostics/best_model.zip --speed 4
```
