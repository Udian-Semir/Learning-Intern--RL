# Sai0-VLA System Architecture Design

This document maps the end-to-end Sai0-VLA system, the training and inference data flow, and the internal algorithm/data flow of the recommended `Flow_Matching_1` action head.

Main path:

```text
VLMs/S0_1 + VLAs/Sai0_1 + Action_Heads/Flow_Matching_1
```

`Flow_Matching_0`, `OFT1_0`, and `ParaCAT` are alternative implementations of the Action Head slot.

---

## 1. System Goal

Input:

```text
multi-view images + language instruction + current robot state
```

Output:

```text
future H-step action chunk: (H, action_dim)
```

Core idea:

```text
Frozen VLM extracts visual-language semantics.
Trainable Action Head decodes VLM hidden states + robot state into robot actions.
Sai0_1 wraps both sides into one training/inference interface.
```

---

## 2. System-Level Module View

```mermaid
flowchart LR
    subgraph Env[Environment / Client]
        IMG[Multi-view images]
        TXT[Instruction]
        STATE[Robot state]
    end

    subgraph Deploy[deployment/Sai0_1_server]
        API[FastAPI /predict /v1/act]
        PRE[Request and image preprocessing]
        RT[RealtimeInference]
    end

    subgraph Orchestrator[VLAs/Sai0_1]
        CFG[Sai0Config]
        MODEL[Sai0Model]
        DU[data_utils / collate]
    end

    subgraph VLM[VLM Backbone: VLMs/S0_1]
        SELECT[create_vlm_backbone]
        QWEN[Qwen3-VL]
        EAGLE[Eagle 2.5 VL]
        COSMOS[Cosmos Reason 2B]
        HS[hidden_states]
    end

    subgraph AH[Action Head Slot]
        FM1[Flow_Matching_1]
        FM0[Flow_Matching_0]
        OFT[OFT1_0]
        PC[ParaCAT]
    end

    subgraph Robot[Execution]
        ACT[action chunk]
        EXEC[execute first steps / closed-loop replan]
    end

    IMG --> API
    TXT --> API
    STATE --> API
    API --> PRE --> RT --> MODEL
    CFG --> MODEL
    MODEL --> SELECT
    SELECT --> QWEN --> HS
    SELECT --> EAGLE --> HS
    SELECT --> COSMOS --> HS
    HS --> MODEL
    STATE --> MODEL
    MODEL --> FM1
    MODEL -.replaceable.-> FM0
    MODEL -.replaceable.-> OFT
    MODEL -.replaceable.-> PC
    FM1 --> ACT --> EXEC
```

---

## 3. Training Data Flow

Training usually uses pre-extracted VLM hidden states. The VLM is not repeatedly run during Action Head training.

```mermaid
flowchart TD
    RAW[Raw demos: image/state/action/instruction]
    LERO[Convert to LeRobot dataset]
    EXT[Offline VLM hidden-state extraction]
    CACHE[vlm_hidden_states/*.npy or chunk-*.npz]
    LOADER[LeRobotDataset + sai0_collate_fn]

    subgraph Batch[Training batch]
        BF[backbone_output<br/>backbone_features: B,S,Dv<br/>backbone_attention_mask: B,S]
        AHIN[action_head_inputs<br/>state: B,1,max_state_dim<br/>action: B,H,max_action_dim<br/>action_mask: B,H,max_action_dim<br/>embodiment_id: B]
    end

    FM[Flow_Matching_1.forward]
    LOSS[masked MSE loss]
    OPT[optimizer / DDP]

    RAW --> LERO --> EXT --> CACHE --> LOADER
    LOADER --> BF
    LOADER --> AHIN
    BF --> FM
    AHIN --> FM
    FM --> LOSS --> OPT
```

The `action` field in the training batch is the expert action from demonstrations. The collate function slices it into a future `H`-step action chunk and pads it to `max_action_dim`.

---

## 4. Realtime Inference Flow

Realtime inference does not use cached hidden states. `Sai0Model.predict` calls the VLM online.

```mermaid
sequenceDiagram
    participant Client as Client / Robot Env
    participant Server as FastAPI Server
    participant RT as RealtimeInference
    participant M as Sai0Model
    participant V as VLM Backbone
    participant A as Flow_Matching_1

    Client->>Server: images + instruction + state
    Server->>Server: auth / rate limit / image preprocess
    Server->>RT: predict(...)
    RT->>M: predict(images, instruction, state)
    M->>V: get_hidden_states(images, instruction)
    V-->>M: List[hidden_states]
    M->>M: build backbone_output + pad state
    M->>A: get_action(backbone_output, action_input)
    A-->>M: action_pred: B,H,action_dim
    M-->>RT: actions
    RT-->>Server: actions + latency
    Server-->>Client: JSON response
```

---

## 5. Key Data Structures

| Name | Typical shape | Source | Meaning |
|---|---:|---|---|
| `images` | multi-view PIL/array | env or dataset | agentview, wrist, etc. |
| `instruction` | string | task | natural-language task instruction |
| `vlm_hidden_states` | `(B,L,S,Dv)` or list | VLM | visual-language tokens from selected layers |
| `backbone_features` | `(B,S,Dv)` | collate or online VLM | layer/merged features consumed by Action Head |
| `state` | `(B,1,max_state_dim)` | dataset/env | current robot proprio state after padding |
| `action` | `(B,H,max_action_dim)` | dataset | expert action chunk, only used in training |
| `action_mask` | `(B,H,max_action_dim)` | collate | masks real action dims and ignores padding |
| `embodiment_id` | `(B,)` | collate/config | selects embodiment-specific parameters |
| `action_pred` | `(B,H,action_dim)` | Action Head | predicted action chunk |

---

## 6. `Flow_Matching_1` Internal Overview

Core files:

```text
Action_Heads/Flow_Matching_1/models/action_head/flow_matching_action_head.py
Action_Heads/Flow_Matching_1/models/action_head/cross_attention_dit.py
```

```mermaid
flowchart TD
    BF[backbone_features B,S,Dv]
    MASK[backbone_attention_mask B,S]
    STATE[state B,1,max_state_dim]
    ACT[action or noisy action B,H,action_dim]
    T[timestep B]
    EID[embodiment_id B]

    VLLN[vlln LayerNorm]
    VLSELF[VL SelfAttentionTransformer]
    SE[state_encoder<br/>CategorySpecificMLP]
    AE[action_encoder<br/>MultiEmbodimentActionEncoder]
    FT[learned future_tokens]
    CAT[concat on sequence dim<br/>state + future + action]
    DIT[DiT<br/>cross-attn to VLM features]
    DEC[action_decoder<br/>CategorySpecificMLP]
    SLICE[slice last H tokens]
    VEL[pred_velocity]

    BF --> VLLN --> VLSELF --> DIT
    MASK --> DIT
    STATE --> SE --> CAT
    ACT --> AE --> CAT
    T --> AE
    EID --> SE
    EID --> AE
    FT --> CAT
    CAT --> DIT --> DEC --> SLICE --> VEL
    EID --> DEC
```

---

## 7. Flow Matching Algorithm Flow

### 7.1 Training Algorithm

```mermaid
flowchart TD
    EXP[expert action]
    NOISE[Gaussian noise]
    TS[sample t from transformed Beta]
    XT[x_t = 1-t noise + t action]
    TARGET[target velocity = action - noise]
    MODEL[Flow_Matching_1.forward]
    PRED[pred_velocity]
    MASK[action_mask]
    LOSS[masked MSE]

    EXP --> XT
    NOISE --> XT
    TS --> XT
    EXP --> TARGET
    NOISE --> TARGET
    XT --> MODEL --> PRED --> LOSS
    TARGET --> LOSS
    MASK --> LOSS
```

Linear flow means the path from random action noise to expert action is a straight interpolation:

```text
x_t = (1 - t) * noise + t * action
velocity = d x_t / dt = action - noise
```

The model learns this conditional velocity field:

```text
v_theta(x_t, t, VLM hidden states, state) ~= action - noise
```

### 7.2 Inference Algorithm

```mermaid
flowchart TD
    INIT[x0 = randn B,H,action_dim]
    LOOP{for step = 0..N-1}
    ENC[encode current action x_t + timestep]
    DIT[DiT predicts velocity]
    EULER[x = x + dt * velocity]
    OUT[action_pred]

    INIT --> LOOP
    LOOP --> ENC --> DIT --> EULER --> LOOP
    LOOP --> OUT
```

Default `num_inference_timesteps=4`, so inference performs four Euler updates.

---

## 8. Token-Level Data Flow

The central path is:

```text
state tokens + future tokens + action tokens
  -> DiT cross-attend VLM hidden states
  -> action_decoder
  -> pred_velocity
```

### 8.1 Token Composition

```mermaid
flowchart LR
    SRAW[raw state<br/>B,1,max_state_dim]
    ARAW[current action x_t<br/>B,H,action_dim]
    TIME[timestep t<br/>B]
    FTAB[learned embedding table<br/>num_future_tokens,D]

    SE[state_encoder]
    AE[action_encoder]
    FEXP[expand to batch]
    STOK[state tokens<br/>B,Ss,Da]
    ATOK[action tokens<br/>B,H,Da]
    FTOK[future tokens<br/>B,F,Da]
    SEQ[sa_embs<br/>B,Ss+F+H,Da]

    SRAW --> SE --> STOK
    ARAW --> AE --> ATOK
    TIME --> AE
    FTAB --> FEXP --> FTOK
    STOK --> SEQ
    FTOK --> SEQ
    ATOK --> SEQ
```

| Token | Source | Role |
|---|---|---|
| `state tokens` | `state_encoder(state, embodiment_id)` | robot body context: gripper, pose, joints, etc. |
| `future tokens` | learned embedding | planning workspace for future context and intermediate computation |
| `action tokens` | `action_encoder(x_t, t, embodiment_id)` | current action chunk being refined, one token per future step |

### 8.2 `state_encoder`

`state_encoder` is a `CategorySpecificMLP`:

```text
state: B,1,max_state_dim
embodiment_id: B
  -> select W/b for this embodiment
  -> Linear(max_state_dim -> hidden_size)
  -> ReLU
  -> Linear(hidden_size -> input_embedding_dim)
  -> state_features: B,1,input_embedding_dim
```

`CategorySpecific` means different robots can use different linear parameters selected by `embodiment_id`.

### 8.3 `action_encoder`

`action_encoder` is a `MultiEmbodimentActionEncoder`:

```text
x_t: B,H,action_dim
t: B
embodiment_id: B
  -> W1: action_dim -> input_embedding_dim
  -> sinusoidal timestep embedding: B,H,input_embedding_dim
  -> concat(action_emb, time_emb): B,H,2*input_embedding_dim
  -> W2 + swish
  -> W3
  -> action_features: B,H,input_embedding_dim
```

Each action token therefore encodes the current action value, the flow time, and the robot embodiment.

### 8.4 DiT Cross-Attention

```mermaid
flowchart TD
    QUERY[Query<br/>state/future/action tokens]
    KV[Key/Value<br/>VLM hidden states]
    T[timestep embedding]
    NORM[AdaLayerNorm with timestep]
    ATTN[Attention]
    FF[FeedForward]
    RES[residual outputs]

    QUERY --> NORM
    T --> NORM
    NORM --> ATTN
    KV --> ATTN
    ATTN --> RES
    RES --> FF --> RES
```

Inside DiT:

```text
hidden_states = state/future/action tokens
encoder_hidden_states = VLM hidden states
timestep = flow time
```

Cross-attention means action-side tokens query the visual-language tokens. Action tokens can attend to target object locations, state tokens can attend to the relation between robot pose and the scene, and future tokens provide extra planning capacity.

DiT output has the same sequence length:

```text
model_output: B,Ss+F+H,hidden_size
```

### 8.5 `action_decoder` and `pred_velocity`

```mermaid
flowchart LR
    MO[model_output<br/>B,Ss+F+H,hidden_size]
    DEC[action_decoder<br/>CategorySpecificMLP]
    PRED[pred<br/>B,Ss+F+H,action_dim]
    LAST[slice last H tokens]
    VEL[pred_velocity<br/>B,H,action_dim]

    MO --> DEC --> PRED --> LAST --> VEL
```

The concatenation order is:

```text
[state tokens][future tokens][action tokens]
```

So the last `H` tokens correspond to the future action chunk. The head keeps only those outputs:

```text
pred_velocity = pred[:, -action_horizon:]
```

---

## 9. Module Responsibility Table

| Module | File | Input | Output | Responsibility |
|---|---|---|---|---|
| VLM factory | `VLMs/S0_1/backbone/model_selector.py` | model type/path/layers | VLM backbone | load Qwen/Eagle/Cosmos by config |
| VLM backbone | `VLMs/S0_1/backbone/*/backbone.py` | images + instruction | hidden states list | extract visual-language tokens |
| Sai0Model | `VLAs/Sai0_1/sai0_model.py` | config / hidden states / state | loss or actions | orchestrate VLM and Action Head |
| data collate | `VLAs/Sai0_1/data_utils.py` | LeRobot samples | BatchFeature pair | normalize, pad, mask, batch |
| FlowMatching Head | `Action_Heads/Flow_Matching_1/.../flow_matching_action_head.py` | backbone_output + action_input | loss or action_pred | learn action velocity field |
| DiT | `Action_Heads/Flow_Matching_1/.../cross_attention_dit.py` | state/future/action tokens + VLM tokens + t | updated tokens | conditional Transformer action modeling |
| deployment server | `deployment/Sai0_1_server/server.py` | HTTP JSON/images | JSON actions | served inference, auth, rate limit, queue |

---

## 10. Replaceable Action Head Boundary

```mermaid
flowchart LR
    INPUT[Sai0Model normalized interface<br/>VLM hidden states + state]
    SLOT[Action Head slot]
    FM1[Flow_Matching_1<br/>velocity field + Euler]
    FM0[Flow_Matching_0<br/>GR00T N1.5 compatible]
    OFT[OFT1_0<br/>Transformer + L1/Diffusion]
    PC[ParaCAT<br/>discrete ternary actions]
    OUT[action chunk]

    INPUT --> SLOT
    SLOT --> FM1 --> OUT
    SLOT -.-> FM0 -.-> OUT
    SLOT -.-> OFT -.-> OUT
    SLOT -.-> PC -.-> OUT
```

Stable inference-side interface:

```text
backbone_features + backbone_attention_mask + state + embodiment_id
```

Training adds:

```text
action + action_mask
```

Any new Action Head can plug into `Sai0_1` if it consumes these tensors and returns either `loss` for training or `action_pred` for inference.
