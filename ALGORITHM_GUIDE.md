# RL-Tycoon-Agent: 알고리즘 상세 문서

> **프로젝트**: RL Tycoon – 강화학습 기반 레스토랑 경영 게임  
> **환경**: 관측 77차원 (`float32`), 행동 7개 (`Discrete`)  
> **행동 공간**: `[↑, ↓, ←, →, 상호작용, 대기, 업그레이드]`

---

## 목차

1. [PPO (Proximal Policy Optimization)](#1-ppo-proximal-policy-optimization)
2. [DQN (Deep Q-Network)](#2-dqn-deep-q-network)
3. [A3C (Asynchronous Advantage Actor-Critic)](#3-a3c-asynchronous-advantage-actor-critic)
4. [SAC (Soft Actor-Critic for Discrete Actions)](#4-sac-soft-actor-critic-for-discrete-actions)
5. [Discrete SAC (Quantile + TQC 스타일)](#5-discrete-sac-quantile--tqc-스타일)
6. [DreamerV3 (RSSM + 상상 기반 학습)](#6-dreamerv3-rssm--상상-기반-학습)
7. [Model-Based RL (World Model + MPC)](#7-model-based-rl-world-model--mpc)
8. [MARL (Multi-Agent Self-Play)](#8-marl-multi-agent-self-play)
9. [CrossPlay (교차 알고리즘 대결 학습)](#9-crossplay-교차-알고리즘-대결-학습)
10. [공통 기반: 관측 공간, 보상 체계, 학습 인프라](#10-공통-기반-관측-공간-보상-체계-학습-인프라)

---

## 1. PPO (Proximal Policy Optimization)

### 1.1 개념

PPO는 **On-Policy** 정책 경사 알고리즘으로, TRPO(Trust Region Policy Optimization)의 복잡한 제약 최적화를 **클리핑(clipping)** 메커니즘으로 단순화한 방법이다. 핵심 아이디어는 정책 업데이트 시 기존 정책과 너무 크게 벗어나지 않도록 비Rate를 제한하는 것이다.

### 1.2 핵심 수식

**확률 비율(Probability Ratio)**:

$$r_t(\theta) = \frac{\pi_\theta(a_t | s_t)}{\pi_{\theta_{\text{old}}}(a_t | s_t)}$$

**클리핑된 목적 함수**:

$$L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) \hat{A}_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t \right) \right]$$

여기서 $\epsilon$은 클리핑 범위 (본 코드에서 `clip_range = 0.2`).

**GAE (Generalized Advantage Estimation)**:

$$\hat{A}_t = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}$$

$$\delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**전체 손실**:

$$L = L^{CLIP} - c_1 L^{VF} + c_2 H[\pi_\theta]$$

| 항              | 의미                 | 본 코드 계수      |
| --------------- | -------------------- | ----------------- |
| $L^{CLIP}$      | 정책 손실            | —                 |
| $L^{VF}$        | 가치 함수 손실 (MSE) | `vf_coef = 0.5`   |
| $H[\pi_\theta]$ | 정책 엔트로피 보너스 | `ent_coef = 0.05` |

### 1.3 프로젝트 적용 설정

```
📁 algorithms/ppo/config.json
```

| 파라미터          | 값         | 설명                           |
| ----------------- | ---------- | ------------------------------ |
| `total_timesteps` | 10,000,000 | 총 학습 스텝 (가장 긴 학습)    |
| `n_envs`          | 8          | 병렬 환경 수 (벡터화)          |
| `learning_rate`   | 3e-4       | 초기 학습률 (선형 감소 스케줄) |
| `n_steps`         | 2048       | 업데이트당 수집 스텝           |
| `batch_size`      | 128        | 미니배치 크기                  |
| `n_epochs`        | 10         | 수집 데이터 재사용 횟수        |
| `gamma`           | 0.99       | 할인율                         |
| `gae_lambda`      | 0.95       | GAE $\lambda$                  |
| `clip_range`      | 0.2        | $\epsilon$ (클리핑 범위)       |
| `ent_coef`        | 0.05       | 엔트로피 보너스 계수           |
| `vf_coef`         | 0.5        | 가치 손실 계수                 |
| `max_grad_norm`   | 0.5        | 그래디언트 클리핑 상한         |
| `net_arch`        | [512, 256] | 은닉층 구조                    |
| `activation_fn`   | tanh       | 활성화 함수                    |

**구현**: Stable-Baselines3 `PPO` 래퍼 + `EvalCallback` + `KoreanEvalStopCallback` (조기 종료 patience=150)

**학습률 스케줄**: 선형 감소 (`lr_schedule: "linear"`)

$$\text{lr}(t) = \text{lr}_0 \times (1 - \tfrac{t}{T})$$

---

## 2. DQN (Deep Q-Network)

### 2.1 개념

DQN은 **Off-Policy** 가치 기반 알고리즘으로, Q-함수를 신경망으로 근사한다. 경험 리플레이(Experience Replay)와 타겟 네트워크(Target Network)라는 두 가지 핵심 트릭으로 학습을 안정화한다.

### 2.2 핵심 수식

**Q-러닝 업데이트 타겟**:

$$y_t = r_t + \gamma \max_{a'} Q_{\theta^-}(s_{t+1}, a')$$

여기서 $\theta^-$는 타겟 네트워크 파라미터.

**손실 함수 (Huber Loss)**:

$$L(\theta) = \mathbb{E}\left[ \text{Huber}\left( Q_\theta(s_t, a_t) - y_t \right) \right]$$

**$\epsilon$-Greedy 탐색 전략**:

$$a_t = \begin{cases} \text{random action} & \text{with probability } \epsilon \\ \arg\max_a Q_\theta(s_t, a) & \text{otherwise} \end{cases}$$

$\epsilon$은 `exploration_initial_eps` (1.0) → `exploration_final_eps` (0.1)으로 점진 감소.

**타겟 네트워크 업데이트** (Hard Copy):

$$\theta^- \leftarrow \theta \quad \text{every } N \text{ steps}$$

본 코드: `target_update_interval = 500`, `tau = 1.0` (hard copy).

### 2.3 프로젝트 적용 설정

```
📁 algorithms/dqn/config.json
```

| 파라미터                  | 값         | 설명                               |
| ------------------------- | ---------- | ---------------------------------- |
| `total_timesteps`         | 300,000    | 총 학습 스텝                       |
| `n_envs`                  | 1          | SB3 DQN은 단일 환경만 지원         |
| `learning_rate`           | 5e-4       | 학습률                             |
| `buffer_size`             | 100,000    | 리플레이 버퍼 크기                 |
| `learning_starts`         | 500        | 학습 시작까지의 랜덤 수집 스텝     |
| `batch_size`              | 64         | 미니배치 크기                      |
| `gamma`                   | 0.99       | 할인율                             |
| `tau`                     | 1.0        | 타겟 네트워크 업데이트 비율 (Hard) |
| `target_update_interval`  | 500        | 타겟 네트워크 교체 주기            |
| `train_freq`              | 4          | 환경 스텝당 학습 주기              |
| `exploration_fraction`    | 0.3        | $\epsilon$ 감소 구간 비율          |
| `exploration_initial_eps` | 1.0        | 초기 $\epsilon$                    |
| `exploration_final_eps`   | 0.1        | 최종 $\epsilon$                    |
| `max_grad_norm`           | 10.0       | 그래디언트 클리핑                  |
| `net_arch`                | [256, 256] | 은닉층 구조                        |
| `activation_fn`           | relu       | 활성화 함수                        |

**구현**: Stable-Baselines3 `DQN` 래퍼

---

## 3. A3C (Asynchronous Advantage Actor-Critic)

### 3.1 개념

A3C는 **비동기(Asynchronous)** Actor-Critic 알고리즘이다. 여러 워커(Worker)가 독립적으로 환경과 상호작용하며 경험을 수집하고, 공유된 글로벌 네트워크에 비동기적으로 그래디언트를 전파한다. 이를 통해 경험 간 상관관계를 자연스럽게 깨뜨린다.

### 3.2 핵심 수식

**Actor-Critic 네트워크**: 공유 특징 추출기 + 정책 헤드($\pi$) + 가치 헤드($V$)

$$\pi(a|s; \theta), \quad V(s; \theta_v)$$

**n-step 리턴**:

$$R_t^{(n)} = \sum_{k=0}^{n-1} \gamma^k r_{t+k} + \gamma^n V(s_{t+n})$$

**Advantage** (GAE 사용 시):

$$\hat{A}_t = \sum_{l=0}^{n-1} (\gamma \lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)$$

**정책 손실**:

$$L_\pi = -\mathbb{E}_t \left[ \log \pi(a_t | s_t) \cdot \hat{A}_t \right]$$

**가치 손실**:

$$L_V = \mathbb{E}_t \left[ (R_t^{(n)} - V(s_t))^2 \right]$$

**전체 손실**:

$$L = L_\pi + c_v L_V - c_e H[\pi]$$

| 항              | 계수  | 본 코드 값              |
| --------------- | ----- | ----------------------- |
| 가치 손실       | $c_v$ | `value_loss_coef = 0.5` |
| 엔트로피 보너스 | $c_e$ | `entropy_coef = 0.01`   |

### 3.3 프로젝트 적용 설정

```
📁 algorithms/a3c/config.json
```

| 파라미터          | 값         | 설명                                      |
| ----------------- | ---------- | ----------------------------------------- |
| `total_timesteps` | 300,000    | 총 학습 스텝                              |
| `n_workers`       | 8          | 비동기 워커 수 (GPU 모드: n-step rollout) |
| `learning_rate`   | 3e-4       | 학습률                                    |
| `gamma`           | 0.99       | 할인율                                    |
| `gae_lambda`      | 0.95       | GAE $\lambda$                             |
| `entropy_coef`    | 0.01       | 엔트로피 계수                             |
| `value_loss_coef` | 0.5        | 가치 손실 계수                            |
| `max_grad_norm`   | 40.0       | 그래디언트 클리핑 (높은 값)               |
| `n_steps`         | 20         | n-step 리턴 길이                          |
| `net_arch`        | [128, 128] | 은닉층 구조                               |
| `activation_fn`   | relu       | 활성화 함수                               |

**구현 특징**:

- **GPU 모드**: 단일 프로세스, 멀티 환경 vectorized rollout
- **CPU 모드**: `torch.multiprocessing`으로 워커별 독립 환경, `SharedAdam` 옵티마이저
- `ActorCritic` 네트워크: 2층 공유 → 정책 헤드(softmax) + 가치 헤드(스칼라)

---

## 4. SAC (Soft Actor-Critic for Discrete Actions)

### 4.1 개념

SAC는 **Maximum Entropy** 강화학습 프레임워크이다. 보상 최대화뿐 아니라 정책의 **엔트로피**를 최대화하여 탐색(exploration)과 활용(exploitation)을 자동으로 균형 맞춘다. 원래 연속 행동 공간용이지만, 본 구현에서는 이산 행동 공간에 맞게 수정하였다.

### 4.2 핵심 수식

**Maximum Entropy 목적 함수**:

$$\pi^* = \arg\max_\pi \mathbb{E}\left[ \sum_t \gamma^t \left( r_t + \alpha H[\pi(\cdot | s_t)] \right) \right]$$

여기서 $\alpha$는 온도(temperature) 파라미터.

**Soft Q-값 (이산 행동)**:

$$Q(s, a): \text{Twin Q-Networks}, \quad Q_1, Q_2$$

**Q-러닝 타겟**:

$$y_t = r_t + \gamma \sum_{a'} \pi(a' | s_{t+1}) \left[ \min(Q_1^-(s_{t+1}, a'), Q_2^-(s_{t+1}, a')) - \alpha \log \pi(a' | s_{t+1}) \right]$$

**정책 손실 (이산)**:

$$L_\pi = \mathbb{E}_s \left[ \sum_a \pi(a|s) \left( \alpha \log \pi(a|s) - \min(Q_1(s,a), Q_2(s,a)) \right) \right]$$

**자동 온도 튜닝**:

$$L_\alpha = -\alpha \mathbb{E}_s \left[ \sum_a \pi(a|s) \log \pi(a|s) + \bar{H} \right]$$

여기서 $\bar{H}$는 목표 엔트로피. 본 코드:

$$\bar{H} = 0.45 \times \log(|\mathcal{A}|) = 0.45 \times \log(7) \approx 0.875$$

**타겟 네트워크 소프트 업데이트**:

$$\theta^- \leftarrow \tau \theta + (1 - \tau) \theta^-$$

### 4.3 프로젝트 적용 설정

```
📁 algorithms/sac/config.json
```

| 파라미터          | 값         | 설명                                 |
| ----------------- | ---------- | ------------------------------------ |
| `total_timesteps` | 1,000,000  | 총 학습 스텝                         |
| `n_envs`          | 4          | 라운드 로빈 수집 환경 수             |
| `learning_rate`   | 3e-4       | 학습률 (Q, Policy, Alpha 모두)       |
| `buffer_size`     | 500,000    | 리플레이 버퍼 크기                   |
| `learning_starts` | 20,000     | 학습 시작 전 랜덤 수집               |
| `batch_size`      | 256        | 미니배치 크기                        |
| `gamma`           | 0.99       | 할인율                               |
| `tau`             | 0.005      | 소프트 업데이트 비율                 |
| `gradient_steps`  | 4          | 환경 스텝당 그래디언트 업데이트 횟수 |
| `max_grad_norm`   | 1.0        | 그래디언트 클리핑                    |
| `target_entropy`  | auto       | $0.45 \times \log(7)$ 자동 계산      |
| `net_arch`        | [256, 256] | 은닉층 구조                          |

**구현 특징**:

- PyTorch 직접 구현 (SB3 SAC는 연속 행동만 지원)
- `SoftQNetwork`: Twin Q (Q1, Q2 분리)
- `PolicyNetwork`: Categorical softmax 정책
- 4환경 라운드 로빈 수집으로 Q-네트워크 과적합 방지
- TensorBoard 로깅: `q_loss`, `policy_loss`, `alpha_loss`, `alpha`, `entropy`

---

## 5. Discrete SAC (Quantile + TQC 스타일)

### 5.1 개념

기본 SAC를 **분포적 Q-학습(Distributional RL)**으로 확장한 버전이다. Q-값의 평균만 학습하는 대신, Q-값의 **전체 분포**를 quantile로 모델링한다. 추가로 TQC(Truncated Quantile Critics) 아이디어를 적용하여 **3개의 Q-네트워크**를 사용하면서 상위 quantile을 드롭해 과대추정을 억제한다.

### 5.2 핵심 수식

**Quantile Regression**:

각 Q-네트워크가 행동 $a$에 대해 $N$개의 quantile $\theta_i(s, a)$를 출력:

$$Q(s, a) = [\theta_1, \theta_2, \ldots, \theta_N]$$

quantile 위치:

$$\tau_i = \frac{2i - 1}{2N}, \quad i = 1, \ldots, N$$

**Quantile Huber Loss**:

$$\rho_\tau^\kappa(\delta) = |\tau - \mathbb{1}(\delta < 0)| \cdot L_\kappa(\delta)$$

$$L_\kappa(\delta) = \begin{cases} \frac{1}{2}\delta^2 & \text{if } |\delta| \leq \kappa \\ \kappa(|\delta| - \frac{1}{2}\kappa) & \text{otherwise} \end{cases}$$

본 코드에서 $\kappa = 1.0$ (Huber threshold).

**TQC – 상위 Quantile 드롭**:

3개의 Q-네트워크에서 총 $3 \times N$ 개의 quantile 값을 모은 후, **상위 $K$개를 제거**하여 과대추정을 억제:

$$\text{sorted quantiles} = \text{sort}([\theta^{(1)}_1, \ldots, \theta^{(3)}_N])$$

$$\hat{Q}(s, a) = \text{mean}(\text{sorted quantiles}[:-K])$$

본 코드: $N = 25$, 3개 네트워크 → 총 75개 quantile 중 **상위 2개 드롭** → 73개 평균.

**정책 학습** (기본 SAC와 동일):

$$L_\pi = \mathbb{E}_s \left[ \sum_a \pi(a|s) \left( \alpha \log \pi(a|s) - \hat{Q}(s, a) \right) \right]$$

### 5.3 프로젝트 적용 설정

```
📁 algorithms/discrete_sac/config.json
```

| 파라미터                | 값         | 설명                                             |
| ----------------------- | ---------- | ------------------------------------------------ |
| `total_timesteps`       | 1,000,000  | 총 학습 스텝                                     |
| `n_envs`                | 4          | 라운드 로빈 수집 환경 수                         |
| `learning_rate`         | 3e-4       | 학습률                                           |
| `buffer_size`           | 500,000    | numpy 배열 기반 리플레이 버퍼                    |
| `learning_starts`       | 20,000     | 학습 시작 전 수집                                |
| `batch_size`            | 256        | 미니배치 크기                                    |
| `gamma`                 | 0.99       | 할인율                                           |
| `tau`                   | 0.005      | 소프트 업데이트 비율                             |
| `gradient_steps`        | 2          | 환경 스텝당 그래디언트 업데이트                  |
| `n_quantiles`           | 25         | Q-네트워크당 quantile 수 $N$                     |
| `n_critics`             | 3          | Q-네트워크 수 (TQC)                              |
| `top_quantiles_to_drop` | 2          | 과대추정 방지를 위해 드롭할 상위 quantile 수 $K$ |
| `target_entropy_ratio`  | 0.45       | 목표 엔트로피 비율                               |
| `max_grad_norm`         | 1.0        | 그래디언트 클리핑                                |
| `net_arch`              | [256, 256] | 은닉층 구조                                      |

**구현 구조**:
| 클래스 | 역할 |
|---|---|
| `QuantileQNetwork` | obs → `[B, A, N_quantiles]` — 행동별 quantile 분포 출력 |
| `PolicyNetwork` | obs → `[B, A]` — softmax 확률 |
| `NumpyReplayBuffer` | 고정 크기 numpy 배열 기반 (deque보다 효율적) |
| `DiscreteSACTrainer` | BaseTrainer 인터페이스, TensorBoard + evaluations.npz |

---

## 6. DreamerV3 (RSSM + 상상 기반 학습)

### 6.1 개념

DreamerV3는 **모델 기반(Model-Based)** 강화학습의 최고봉으로, 환경의 역학을 **세계 모델(World Model)**로 학습한 뒤, 모델 안에서 **상상(Imagination)** 궤적을 생성하여 정책을 학습한다. 실제 환경과 상호작용하는 횟수를 극적으로 줄일 수 있다.

핵심 구조는 **RSSM(Recurrent State-Space Model)**이다:

```
관측 o_t → 인코더 → posterior(h_t, z_post_t)  [현실 보정]
                          ↓
            prior(h_t, z_prior_t)  [모델만의 예측]
                          ↓
         (h_t, z_t) → 디코더, 보상예측, 종료예측
```

### 6.2 핵심 수식

**RSSM 상태 구성**:

$$\text{state}_t = (h_t, z_t), \quad h_t \in \mathbb{R}^{256}, \quad z_t \in \{0,1\}^{32 \times 32}$$

- $h_t$: 결정론적(deterministic) 상태 — GRU hidden
- $z_t$: 확률적(stochastic) 상태 — 32개의 Categorical 분포, 각 32 클래스

**Feature 차원**: $\text{dim}(h) + \text{dim}(z) = 256 + 32 \times 32 = 1280$

**시퀀스 모델 (GRU)**:

$$h_t = \text{GRU}\left( \text{ELU}\left( W_{\text{in}} [z_{t-1}, a_{t-1}] \right), \; h_{t-1} \right)$$

**Posterior** (관측을 사용한 보정):

$$q(z_t | h_t, o_t) = \text{Categorical}\left( \text{MLP}([h_t, o_t]) \right)$$

**Prior** (모델만의 예측):

$$p(z_t | h_t) = \text{Categorical}\left( \text{MLP}(h_t) \right)$$

**Straight-Through Gumbel-Softmax 샘플링**:

$$z = \text{one\_hot}(\arg\max) + \text{softmax}(\text{logits}) - \text{sg}(\text{softmax}(\text{logits}))$$

forward에서는 one-hot, backward에서는 softmax의 그래디언트를 사용.

---

#### 세계 모델 학습 손실

$$L_{\text{world}} = L_{\text{obs}} + L_{\text{reward}} + L_{\text{continue}} + L_{\text{KL}}$$

| 손실 항   | 수식                                                                             | 설명                   |
| --------- | -------------------------------------------------------------------------------- | ---------------------- |
| 관측 복원 | $L_{\text{obs}} = \text{MSE}(\hat{o}_t, o_t)$                                    | 디코더가 관측을 복원   |
| 보상 예측 | $L_{\text{reward}} = \text{MSE}(\hat{r}_t, r_t)$                                 | 보상 예측기            |
| 종료 예측 | $L_{\text{continue}} = \text{BCE}(\hat{c}_t, 1-d_t)$                             | 에피소드 계속 확률     |
| KL 발산   | $L_{\text{KL}} = \max\left( D_{\text{KL}}[q \| p], \; \text{free\_nats} \right)$ | posterior ↔ prior 정렬 |

**Free Nats** ($= 1.0$): KL이 너무 빨리 0으로 수렴하는 것을 방지 (최소값 보장).

$$D_{\text{KL}}[q \| p] = \sum_{i=1}^{32} \sum_{c=1}^{32} q_{i,c} \log \frac{q_{i,c}}{p_{i,c}}$$

---

#### 상상 기반 Actor-Critic

실제 환경 대신, 학습된 RSSM 안에서 $H$스텝 상상 궤적을 생성:

$$\text{for } t = 1 \ldots H: \quad a_t \sim \pi(z_t, h_t), \quad (h_{t+1}, z_{t+1}) = \text{imagine\_step}(h_t, z_t, a_t)$$

**Lambda-Return** (TD($\lambda$)):

$$G_t^\lambda = r_t + \gamma c_t \left[ (1-\lambda) V(s_{t+1}) + \lambda G_{t+1}^\lambda \right]$$

$$G_H^\lambda = V(s_H), \quad c_t = \sigma(\hat{c}_t) \approx 1 - \hat{d}_t$$

**Critic 손실**:

$$L_V = \mathbb{E}_t \left[ (V(s_t) - G_t^\lambda)^2 \right]$$

**Actor 손실** (REINFORCE + 엔트로피):

$$L_\pi = -\mathbb{E}_t \left[ \log \pi(a_t | s_t) \cdot (G_t^\lambda - V(s_t)) \right] - c_e H[\pi]$$

### 6.3 프로젝트 적용 설정

```
📁 algorithms/dreamer/config.json
```

| 파라미터                 | 값        | 설명                                     |
| ------------------------ | --------- | ---------------------------------------- |
| `total_timesteps`        | 1,000,000 | 총 학습 스텝                             |
| `n_envs`                 | 1         | 단일 환경 (세계 모델이 데이터 효율 보완) |
| `learning_rate_world`    | 3e-4      | RSSM + 디코더 + 예측기 학습률            |
| `learning_rate_actor`    | 1e-4      | Actor 학습률 (보수적)                    |
| `learning_rate_critic`   | 3e-4      | Critic 학습률                            |
| `buffer_size`            | 500,000   | 시퀀스 리플레이 버퍼                     |
| `batch_size`             | 64        | 배치 크기                                |
| `seq_len`                | 32        | 시퀀스 샘플 길이 $T$                     |
| `gamma`                  | 0.997     | 할인율 (높은 값 → 먼 미래 중시)          |
| `lambda_`                | 0.95      | TD($\lambda$) 파라미터                   |
| `imagination_horizon`    | 15        | 상상 궤적 길이 $H$                       |
| `world_model_train_freq` | 100       | 세계 모델 학습 주기 (100스텝마다)        |
| `entropy_coef`           | 0.003     | 엔트로피 보너스 (매우 작음)              |
| `max_grad_norm`          | 100.0     | 그래디언트 클리핑 (느슨)                 |
| `free_nats`              | 1.0       | KL 발산 하한                             |
| `rssm_hidden`            | 256       | GRU hidden 차원                          |
| `rssm_stochastic`        | 32        | Categorical 분포 개수                    |
| `rssm_discrete_classes`  | 32        | 분포당 클래스 수                         |

**네트워크 모듈 구성**:

| 모듈                | 구조                           | 입/출력                 |
| ------------------- | ------------------------------ | ----------------------- |
| `RSSM`              | GRU + Prior/Posterior MLP      | (h, z, a, o) → (h', z') |
| `ObsDecoder`        | MLP [1280→256→256→77]          | feature → 관측 복원     |
| `RewardPredictor`   | MLP [1280→256→256→1]           | feature → 보상          |
| `ContinuePredictor` | MLP [1280→256→1]               | feature → 종료 확률     |
| `Actor`             | MLP [1280→256→256→7] + softmax | feature → 행동 확률     |
| `Critic`            | MLP [1280→256→256→1]           | feature → 가치          |

---

## 7. Model-Based RL (World Model + MPC)

### 7.1 개념

환경의 전이 함수($s' = f(s, a)$)와 보상 함수($r = g(s, a)$)를 신경망으로 학습하고, 이 모델을 사용해 **MPC(Model Predictive Control)**로 계획(Planning)을 수행한다. **앙상블(Ensemble)** 모델을 사용하여 예측 불확실성을 추정한다.

### 7.2 핵심 수식

**앙상블 세계 모델** ($K = 3$ 모델):

$$f_k(s_t, a_t) = (\hat{s}_{t+1}^{(k)}, \hat{r}_t^{(k)}, \hat{d}_t^{(k)}), \quad k = 1, \ldots, K$$

**잔차 예측(Residual Prediction)**:

$$\hat{s}_{t+1}^{(k)} = W^{(k)} [s_t, \text{onehot}(a_t)] + s_t$$

관측값에 잔차를 더하는 방식 → 학습 안정성 향상.

**세계 모델 손실**:

$$L_{\text{WM}} = \sum_{k=1}^{K} \left[ \text{MSE}(\hat{s}^{(k)}, s') + \text{MSE}(\hat{r}^{(k)}, r) + \text{BCE}(\hat{d}^{(k)}, d) \right]$$

**MPC (Model Predictive Control)**:

주어진 상태 $s_t$에서 $M$개의 랜덤 행동 시퀀스를 시뮬레이션:

$$\text{for } j = 1 \ldots M: \quad a_{t:t+H}^{(j)} \sim \text{Random}, \quad R_j = \sum_{h=0}^{H-1} \gamma^h \hat{r}_{t+h}^{(j)}$$

최적 행동: $a_t^* = a_t^{(\arg\max_j R_j)}$

**불확실성 추정**:

$$\sigma(s') = \text{Std}([\hat{s}'^{(1)}, \ldots, \hat{s}'^{(K)}])$$

### 7.3 프로젝트 적용 설정

```
📁 algorithms/model_based/config.json
```

| 파라미터                     | 값         | 설명                        |
| ---------------------------- | ---------- | --------------------------- |
| `total_timesteps`            | 300,000    | 총 학습 스텝                |
| `world_model_train_freq`     | 1,000      | 세계 모델 학습 주기         |
| `world_model_epochs`         | 10         | 에포크 수 (학습 주기당)     |
| `world_model_batch_size`     | 128        | 배치 크기                   |
| `planning_horizon`           | 10         | MPC 계획 지평 $H$           |
| `num_simulated_trajectories` | 100        | 시뮬레이션 궤적 수 $M$      |
| `real_data_ratio`            | 0.5        | 실제/시뮬레이션 데이터 비율 |
| `buffer_size`                | 100,000    | 전이 버퍼 크기              |
| `learning_rate`              | 3e-4       | 정책 학습률                 |
| `world_model_lr`             | 1e-3       | 세계 모델 학습률            |
| `gamma`                      | 0.99       | 할인율                      |
| `world_model_hidden`         | [256, 256] | 세계 모델 은닉층            |
| `policy_hidden`              | [128, 128] | 정책 네트워크 은닉층        |

**구현 구조**:

| 클래스              | 역할                                               |
| ------------------- | -------------------------------------------------- |
| `WorldModel`        | 3-앙상블 전이/보상/종료 예측 (잔차 + Dropout 0.05) |
| `WorldModelTrainer` | 앙상블 학습 루프                                   |
| `PolicyNet`         | 공유 특징 추출 + Actor/Critic 헤드                 |
| `TransitionBuffer`  | deque 기반 전이 저장                               |
| `MPCPlanner`        | 100개 궤적 × 10스텝 시뮬레이션                     |

---

## 8. MARL (Multi-Agent Self-Play)

### 8.1 개념

**자기 대결(Self-Play)** 방식으로, 에이전트가 자기 자신의 과거 버전과 경쟁하며 학습한다. 두 매장이 동시에 운영되고, 상대방의 성과(돈, 평점, 서빙 수)를 관측에 포함하여 **상대적 전략**을 학습한다.

### 8.2 핵심 수식

**관측 공간 확장**:

$$o_t = [\text{내 매장 관측}_{77}, \; \text{상대 요약}_3] \in \mathbb{R}^{80}$$

상대 요약: $[\text{money\_ratio}, \;\text{rating}, \;\text{served\_ratio}]$

**상대적 보상**:

$$r_t = r_{\text{base}} + r_{\text{shaping}} + r_{\text{relative}}$$

$$r_{\text{relative}} = c_m \cdot \frac{\text{my\_money} - \text{opp\_money}}{\text{target\_money}} + c_r \cdot (\text{my\_rating} - \text{opp\_rating})$$

| 계수  | 값  | 의미             |
| ----- | --- | ---------------- |
| $c_m$ | 0.5 | 돈 차이 보너스   |
| $c_r$ | 1.0 | 평점 차이 보너스 |

**ELO 레이팅**:

$$E_A = \frac{1}{1 + 10^{(R_B - R_A)/400}}, \quad R'_A = R_A + K(S_A - E_A)$$

$K = 32$, 초기 ELO $= 1000$.

**상대 풀(Opponent Pool)**:

| 파라미터                | 값                                  |
| ----------------------- | ----------------------------------- |
| `self_play_update_freq` | 10,000 스텝마다 풀에 현재 모델 추가 |
| `opponent_pool_size`    | 최대 5개 과거 버전 보관             |

### 8.3 프로젝트 적용 설정

```
📁 algorithms/marl/config.json
```

| 파라미터          | 값         | 설명            |
| ----------------- | ---------- | --------------- |
| `total_timesteps` | 300,000    | 총 학습 스텝    |
| `n_envs`          | 2          | SelfPlayEnv 2개 |
| `learning_rate`   | 3e-4       | PPO 학습률      |
| `n_steps`         | 2048       | 업데이트 주기   |
| `batch_size`      | 128        | 미니배치        |
| `n_epochs`        | 10         | 에포크          |
| `gamma`           | 0.99       | 할인율          |
| `clip_range`      | 0.2        | PPO 클리핑      |
| `ent_coef`        | 0.01       | 엔트로피 계수   |
| `net_arch`        | [128, 128] | 은닉층          |
| `elo_initial`     | 1000       | 초기 ELO        |
| `elo_k`           | 32         | ELO K-인수      |

**구현**: SB3 PPO + `SelfPlayEnv` + `SelfPlayCallback` (주기적 풀 업데이트)

---

## 9. CrossPlay (교차 알고리즘 대결 학습)

### 9.1 개념

MARL이 **자기 자신의 과거 버전**과 싸우는 것과 달리, CrossPlay는 **서로 다른 알고리즘으로 학습된 모델들**을 상대 풀에 넣고 경쟁한다. PPO로 학습된 모델, SAC로 학습된 모델 등이 모두 상대가 될 수 있다.

### 9.2 핵심 메커니즘

**상대 풀 구축**:

```
models/ 디렉토리 자동 스캔 → best_model.zip, best_model.pt, final_model.pt 탐색
          ↓
train_config_used.json에서 알고리즘 이름 탐지
          ↓
load_agent()로 각 모델 로드 → 상대 풀에 추가
```

**라운드 로빈 상대 교체**:

$$\text{opponent\_idx} = \left\lfloor \frac{\text{step}}{\text{swap\_freq}} \right\rfloor \mod |\text{pool}|$$

`swap_freq = 10,000` 스텝마다 상대 교체.

**학습 알고리즘**: PPO (SB3) — `SelfPlayEnv`를 사용하되, 상대가 다른 알고리즘의 에이전트.

### 9.3 프로젝트 적용 설정

```
📁 algorithms/cross_play/config.json
```

| 파라미터                | 값         | 설명                  |
| ----------------------- | ---------- | --------------------- |
| `total_timesteps`       | 200,000    | 총 학습 스텝          |
| `n_envs`                | 4          | 병렬 SelfPlayEnv      |
| `opponent_swap_freq`    | 10,000     | 상대 교체 주기        |
| `learning_rate`         | 3e-4       | PPO 학습률            |
| `n_steps`               | 2048       | 업데이트 주기         |
| `batch_size`            | 64         | 미니배치              |
| `clip_range`            | 0.2        | 클리핑                |
| `ent_coef`              | 0.01       | 엔트로피              |
| `net_arch`              | [256, 256] | 은닉층                |
| `relative_money_bonus`  | 0.5        | 상대 대비 돈 보너스   |
| `relative_rating_bonus` | 1.0        | 상대 대비 평점 보너스 |

---

## 10. 공통 기반: 관측 공간, 보상 체계, 학습 인프라

### 10.1 관측 공간 (77차원)

| 인덱스 | 차원 | 내용                                                     |
| ------ | ---- | -------------------------------------------------------- |
| 0–3    | 4    | 플레이어 (x, y, facing, 속도)                            |
| 4–5    | 2    | 운반 상태 (has_food, has_drink)                          |
| 6–9    | 4    | 이동 가능 여부 (상하좌우)                                |
| 10–57  | 48   | 테이블 상태 (8테이블 × 6: 점유/주문/음식/음료/상태/거리) |
| 58–60  | 3    | 주방 (ready, 대기주문, 거리)                             |
| 61–66  | 6    | 랜드마크 거리 (주방, 카운터, 쓰레기통 등)                |
| 67–68  | 2    | 타겟 방향 (dx, dy 정규화)                                |
| 69–76  | 8    | 게임 상태 (돈, 목표, 평점, 시간, 날짜 등)                |

### 10.2 보상 체계

**서비스 체인** (고객 1명 서빙 시 총 ~31점):

| 이벤트           | 보상 | 설명             |
| ---------------- | ---- | ---------------- |
| `take_order`     | +6.0 | 주문 받기        |
| `submit_kitchen` | +5.0 | 주방에 주문 전달 |
| `pickup_food`    | +5.0 | 음식 픽업        |
| `serve_food`     | +5.0 | 음식 서빙        |
| `pickup_drink`   | +5.0 | 음료 픽업        |
| `serve_drink`    | +5.0 | 음료 서빙        |

**페널티**:

| 이벤트                  | 보상  | 설명                   |
| ----------------------- | ----- | ---------------------- |
| `lost_customer`         | -15.0 | 고객 이탈              |
| `wrong_table`           | -1.5  | 잘못된 테이블 서빙     |
| `trash`                 | -4.0  | 쓰레기통에 음식 버리기 |
| `idle_penalty`          | -1.3  | 매 대기 스텝           |
| `time_penalty`          | -0.02 | 시간 경과 페널티       |
| `blocked_move`          | -0.2  | 벽 충돌                |
| `customer_waiting`      | -0.3  | 고객 대기 중           |
| `waiting_customer_left` | -8.0  | 대기 고객 이탈         |

**전략적 보상**:

| 이벤트              | 보상   | 설명             |
| ------------------- | ------ | ---------------- |
| `buy_upgrade`       | +5.0   | 업그레이드 구매  |
| `win`               | +200.0 | 목표 달성        |
| `net_profit_delta`  | ×0.2   | 순이익 변화      |
| `rating_delta`      | ×10.0  | 평점 변화        |
| `final_score_delta` | ×0.05  | 최종 스코어 변화 |

### 10.3 Dense Shaping (포텐셜 기반)

모든 환경에서 **포텐셜 기반 보상 형성(Potential-Based Reward Shaping)**을 사용:

$$F(s, s') = \gamma \Phi(s') - \Phi(s)$$

$$\Phi(s) = -\frac{\text{dist}(\text{player}, \text{target})}{\text{map\_diagonal}}$$

스케일 3.0, 클리핑 $\pm 1.0$. 에이전트를 올바른 목표(서빙 대상, 주방 등)로 유도.

### 10.4 학습 인프라

| 기능                 | 구현                                                                      |
| -------------------- | ------------------------------------------------------------------------- |
| **TensorBoard 로깅** | 모든 9개 알고리즘에서 `rollout/`, `train/`, `eval/` 지표 기록             |
| **조기 종료**        | `EarlyStopTracker(patience=50, min_delta=1.0)` — 50회 평가 미개선 시 종료 |
| **평가 로그**        | `evaluations.npz` — timesteps + results 저장                              |
| **설정 저장**        | `train_config_used.json` — 사용된 전체 설정 기록                          |
| **자동 버전**        | `_next_version_path()` — `models/ppo` → `models/ppo_v2` 자동 증가         |
| **벤치마크**         | `--benchmark` — 모든 알고리즘 순차 학습 + 결과 비교                       |
| **토너먼트**         | `--mode tournament` — 학습된 모델들의 실시간 경쟁 관전                    |

### 10.5 알고리즘 비교 요약

| 알고리즘        | 유형                        | 데이터 효율 | 학습 안정성 | 핵심 강점                   |
| --------------- | --------------------------- | ----------- | ----------- | --------------------------- |
| **PPO**         | On-Policy                   | 중간        | ★★★★★       | 범용, 안정적, 병렬화 용이   |
| **DQN**         | Off-Policy (Value)          | 높음        | ★★★☆☆       | 단순, 이산 행동에 특화      |
| **A3C**         | Async On-Policy             | 중간        | ★★★☆☆       | 비동기 병렬화, 빠른 수렴    |
| **SAC**         | Off-Policy (MaxEnt)         | 높음        | ★★★★☆       | 자동 탐색-활용 균형         |
| **DiscreteSAC** | Off-Policy (Distributional) | 높음        | ★★★★★       | 분포 학습, 과대추정 억제    |
| **DreamerV3**   | Model-Based                 | ★★★★★       | ★★★☆☆       | 상상 학습, 최고 데이터 효율 |
| **ModelBased**  | Model-Based + MPC           | ★★★★☆       | ★★★☆☆       | 앙상블 불확실성, 계획 수립  |
| **MARL**        | Self-Play                   | 중간        | ★★★★☆       | 자기 대결, ELO 레이팅       |
| **CrossPlay**   | Cross-Algorithm             | 중간        | ★★★★☆       | 다양한 전략 대응, 범용성    |

---

## 부록: 실행 명령어

```bash
# 개별 알고리즘 학습
python -m algorithms.train_launcher --algo PPO --timesteps 10000000
python -m algorithms.train_launcher --algo DQN --timesteps 300000
python -m algorithms.train_launcher --algo A3C --timesteps 300000
python -m algorithms.train_launcher --algo SAC --timesteps 1000000
python -m algorithms.train_launcher --algo DiscreteSAC --timesteps 1000000
python -m algorithms.train_launcher --algo Dreamer --timesteps 1000000
python -m algorithms.train_launcher --algo ModelBased --timesteps 300000
python -m algorithms.train_launcher --algo MARL --timesteps 300000
python -m algorithms.train_launcher --algo CrossPlay --timesteps 200000

# 전체 벤치마크
python -m algorithms.train_launcher --benchmark --timesteps 100000

# 토너먼트 관전
python main.py --mode tournament --speed 3

# TensorBoard 모니터링
tensorboard --logdir models/
```
