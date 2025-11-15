# CAPTCHA Recognition Pipeline Architecture

**Model**: CTC-CRNN (v10 - Final)  
**Performance**: 55.6% sequence accuracy, 85.8% character accuracy  

---

## Complete Pipeline Overview

```mermaid
flowchart TB
    subgraph Input["Input"]
        A[CAPTCHA Image<br/>Variable size, RGB]
    end
    
    subgraph Preprocessing["🔧 Preprocessing"]
        B[Remove Black Lines<br/>HSV masking + inpainting]
        C[Convert to Grayscale]
        D{Training Mode?}
        E[Data Augmentation<br/>rotation, shear,<br/>brightness, noise, fake lines]
        F[Resize & Pad<br/>80 x 280 pixels]
        G[Normalize to 0-1<br/>Add channel dimension]
    end
    
    subgraph Model["CRNN Model"]
        H[ResNet CNN<br/>Feature Extraction]
        I[Sequence Mapping<br/>Reshape to time series]
        J[BiLSTM<br/>Sequence Modeling]
        K[Fully Connected<br/>Character Classification]
        L[Log Softmax<br/>Probability Distribution]
    end
    
    subgraph Output["Output"]
        M[CTC Greedy Decode]
        N[Predicted Text<br/>4-8 characters]
    end
    
    subgraph Training["Training Loop<br/>per batch"]
        O[CTC Loss + Label Smoothing]
        P[Backprop + Grad Clip]
        Q[AdamW Optimizer Step]
        R{End of Epoch?}
        S[LR Schedule:<br/>Warmup → ReduceLROnPlateau]
        T[Early Stopping Check<br/>patience=10]
    end
    
    A --> B --> C --> D
    D -->|Yes| E --> F
    D -->|No| F
    F --> G --> H --> I --> J --> K --> L
    
    L -->|Inference| M --> N
    L -->|Training| O --> P --> Q --> R
    R -->|Yes| S --> T
    R -->|No| L
    
    style A fill:#e1f5ff
    style N fill:#d4edda
    style H fill:#fff3cd
    style J fill:#fff3cd
    style O fill:#f8d7da
    style E fill:#ffe6cc
```

---

## Detailed Model Architecture (v10)

```mermaid
graph TB
    subgraph Input["Input Layer"]
        I1[Image: 1 x 80 x 280<br/>Grayscale, normalized]
    end
    
    subgraph CNN["ResNet-style CNN Feature Extractor"]
        direction TB
        C1["Conv2d(1→64, 3x3)<br/>+ BatchNorm + ReLU"]
        P1["MaxPool(2x2)<br/>→ 64 x 40 x 140"]
        D1["Dropout2d(0.15)"]
        
        R1["ResBlock 1a (64→128)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        R2["ResBlock 1b (128→128)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        P2["MaxPool(2x2)<br/>→ 128 x 20 x 70"]
        D2["Dropout2d(0.15)"]
        
        R3["ResBlock 2a (128→256)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        R4["ResBlock 2b (256→256)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        P3["MaxPool(2x1)<br/>→ 256 x 10 x 70"]
        D3["Dropout2d(0.15)"]
        
        R5["ResBlock 3a (256→512)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        R6["ResBlock 3b (512→512)<br/>Conv-BN-ReLU-Conv-BN + skip"]
        P4["MaxPool(2x1)<br/>→ 512 x 5 x 70"]
        D4["Dropout2d(0.2)"]
    end
    
    subgraph Sequence["Sequence Mapping"]
        S1["Reshape<br/>512 x 5 x 70<br/>→<br/>70 x (512*5)"]
        S2["70 time steps<br/>2560 features each"]
    end
    
    subgraph RNN["Bidirectional LSTM"]
        L1["LSTM Layer 1<br/>input: 2560<br/>hidden: 384<br/>bidirectional"]
        L2["LSTM Layer 2<br/>input: 768<br/>hidden: 384<br/>bidirectional<br/>dropout: 0.3"]
        L3["Output: 70 x 768"]
    end
    
    subgraph Classifier["Classification Head"]
        FC["Fully Connected<br/>768 → 63"]
        LS["Log Softmax"]
        OUT["Output: 70 x 63<br/>70 timesteps<br/>63 classes (a-z, 0-9 + blank)"]
    end
    
    I1 --> C1 --> P1 --> D1
    D1 --> R1 --> R2 --> P2 --> D2
    D2 --> R3 --> R4 --> P3 --> D3
    D3 --> R5 --> R6 --> P4 --> D4
    D4 --> S1 --> S2
    S2 --> L1 --> L2 --> L3
    L3 --> FC --> LS --> OUT
    
    style I1 fill:#e1f5ff
    style OUT fill:#d4edda
    style C1 fill:#fff3cd
    style R1 fill:#ffe4b5
    style R2 fill:#ffe4b5
    style R3 fill:#ffe4b5
    style R4 fill:#ffe4b5
    style R5 fill:#ffe4b5
    style R6 fill:#ffe4b5
    style L1 fill:#ffd4d4
    style L2 fill:#ffd4d4
```

---

## Training Loop Flow

```mermaid
flowchart TD
    Start([Start Training]) --> Init[Initialize Model<br/>HIDDEN_SIZE=384<br/>NUM_LAYERS=2<br/>BATCH_SIZE=64]
    Init --> LoadData[Load Training Data<br/>8,010 images<br/>with augmentation]
    
    LoadData --> Epoch{Epoch Loop<br/>max 150 epochs}
    
    Epoch -->|For each batch| Forward[Forward Pass<br/>Image → CRNN → Log Probs]
    Forward --> CTCLoss[Compute CTC Loss<br/>+ Label Smoothing 0.1]
    CTCLoss --> Backward[Backward Pass<br/>+ Gradient Clipping 5.0]
    Backward --> Update[Update Weights<br/>AdamW lr=0.001<br/>weight_decay=1e-4]
    
    Update --> Val{Validation}
    Val --> Decode[Greedy CTC Decode<br/>Collapse blanks & repeats]
    Decode --> Metrics[Compute Metrics<br/>Loss & Accuracy]
    
    Metrics --> LRSchedule{LR Schedule}
    LRSchedule -->|Epoch < 5| Warmup[Warmup: 0.0001 → 0.001]
    LRSchedule -->|Epoch ≥ 5| Plateau[ReduceLROnPlateau<br/>factor=0.5, patience=8]
    
    Warmup --> EarlyStop{Early Stopping}
    Plateau --> EarlyStop
    
    EarlyStop -->|Val loss improved| Reset[Reset counter] --> SaveBest[Save Best Model]
    EarlyStop -->|No improvement| Counter[Increment counter]
    
    SaveBest --> CheckEpoch{More epochs?}
    Counter --> CheckStop{Counter < 10?}
    
    CheckStop -->|Yes| CheckEpoch
    CheckStop -->|No| Done([Stop: Early Stopping])
    
    CheckEpoch -->|Yes| Epoch
    CheckEpoch -->|No| Done
    
    style Start fill:#d4edda
    style Done fill:#d4edda
    style CTCLoss fill:#f8d7da
    style SaveBest fill:#fff3cd
```

---

## Data Flow Dimensions

```mermaid
graph LR
    subgraph "Input Image"
        I["Variable size<br/>RGB<br/>(H x W x 3)"]
    end
    
    subgraph "Preprocessing"
        P1["80 x 280 x 1<br/>Grayscale"]
    end
    
    subgraph "CNN Stages"
        C1["64 x 40 x 140<br/>After pool1"]
        C2["128 x 20 x 70<br/>After pool2"]
        C3["256 x 10 x 70<br/>After pool3"]
        C4["512 x 5 x 70<br/>After pool4"]
    end
    
    subgraph "Sequence"
        S1["70 x 2560<br/>Time x Features"]
    end
    
    subgraph "LSTM"
        L1["70 x 768<br/>Bidirectional output"]
    end
    
    subgraph "Output"
        O1["70 x 63<br/>Time x Classes"]
    end
    
    I --> P1
    P1 --> C1 --> C2 --> C3 --> C4
    C4 --> S1 --> L1 --> O1
    
    style I fill:#e1f5ff
    style O1 fill:#d4edda
```

---

## ResNet Block Detail

```mermaid
graph TB
    Input["Input<br/>C_in channels"] --> Conv1["Conv2d(C_in → C_out)<br/>3x3, padding=1"]
    Conv1 --> BN1["BatchNorm2d"]
    BN1 --> ReLU1["ReLU"]
    ReLU1 --> Conv2["Conv2d(C_out → C_out)<br/>3x3, padding=1"]
    Conv2 --> BN2["BatchNorm2d"]
    
    Input --> Skip{C_in ≠ C_out?}
    Skip -->|Yes| Down["Conv2d(C_in → C_out)<br/>1x1<br/>+ BatchNorm"]
    Skip -->|No| Identity["Identity"]
    
    Down --> Add["Add (Skip Connection)"]
    Identity --> Add
    BN2 --> Add
    
    Add --> ReLU2["ReLU"]
    ReLU2 --> Output["Output<br/>C_out channels"]
    
    style Input fill:#e1f5ff
    style Output fill:#d4edda
    style Add fill:#fff3cd
```

---

## CTC Loss & Decoding

```mermaid
flowchart TB
    subgraph Model_Output["Model Output"]
        M1["Log Probs<br/>T x N x C<br/>70 x 64 x 63"]
    end
    
    subgraph Training["During Training"]
        T1["Target Labels<br/>Variable length"]
        T2["CTC Loss<br/>Alignment-free"]
        T3["+ Label Smoothing<br/>smoothing=0.1"]
        T4["Final Loss"]
    end
    
    subgraph Inference["During Inference"]
        I1["Greedy Decode<br/>Pick max at each timestep"]
        I2["Collapse Repeats<br/>'hhello' → 'helo'"]
        I3["Remove Blanks<br/>Remove CTC blank token"]
        I4["Final Prediction<br/>e.g. 'q02a9jk'"]
    end
    
    M1 --> T2
    T1 --> T2
    T2 --> T3 --> T4
    
    M1 --> I1 --> I2 --> I3 --> I4
    
    style M1 fill:#e1f5ff
    style T4 fill:#f8d7da
    style I4 fill:#d4edda
```

---

## Augmentation Pipeline

```mermaid
graph TB
    Input["Grayscale Image<br/>80 x 280"] --> Check{Training Mode?}
    
    Check -->|No| Skip["No Augmentation"]
    Check -->|Yes| Aug
    
    subgraph Aug["Augmentation (70% probability each)"]
        A1["Random Rotation<br/>±5 degrees"]
        A2["Random Shear/Skew<br/>±10 degrees"]
        A3["Random Translation<br/>±3, ±2 pixels"]
        A4["Random Black Lines<br/>1-2 lines, thickness 1-2px"]
        A5["Random Brightness<br/>factor 0.85-1.15"]
        A6["Gaussian Noise<br/>mean=0, std=2"]
    end
    
    Aug --> A1 --> A2 --> A3 --> A4 --> A5 --> A6
    
    A6 --> Output["Augmented Image"]
    Skip --> Output
    
    style Input fill:#e1f5ff
    style Output fill:#d4edda
```

---

## Hyperparameters Summary

```mermaid
mindmap
  root((v10 Config))
    Architecture
      CNN: ResNet-style
      LSTM: 384 hidden, 2 layers
      Dropout: 0.15-0.2 CNN, 0.3 LSTM
      Params: ~8M trainable
    Training
      Optimizer: AdamW
      LR: 0.001
      Weight Decay: 1e-4
      Batch Size: 64
      Epochs: 150 max
    Loss
      CTC Loss
      Label Smoothing: 0.1
      Gradient Clip: 5.0
    Schedule
      Warmup: 5 epochs
      ReduceLROnPlateau
      Factor: 0.5
      Patience: 8
    Regularization
      Dropout2d: 0.15-0.2
      LSTM Dropout: 0.3
      Weight Decay: 1e-4
      Early Stop: patience 10
    Augmentation
      Rotation: ±5°
      Shear: ±10°
      Translation: ±3,±2
      Lines: 1-2 random
      Brightness: 0.85-1.15
      Noise: Gaussian std=2
```

---

## Model Evolution Timeline

```mermaid
timeline
    title CRNN Development Journey
    section Early Attempts (v1-v3)
        v1 : 35% : Baseline (overfitting)
        v2 : 22% : Too much regularization
        v3 : 42% : Fixed LR schedule : +7% breakthrough
    section Architecture Changes (v4-v9)
        v4 : 34% : Attention failed
        v5 : 17% : Custom loss catastrophe
        v6 : 42% : Stable baseline
        v7 : 42% : Label smoothing
        v8 : 50% : ResNet breakthrough : +8%
        v9 : 41% : Bigger model failed
    section Best Performance (v10-v12)
        v10 : 55.6% : Enhanced augmentation : +5.6% : BEST
        v11 : 49% : CLAHE/sharpen hurt
        v12 : 54% : Filtering no gain
    section Final Tests (v13-v14)
        v13 : 52% : Multi-scale + curriculum unstable
        v14 : 54.65% : Multi-scale only : +0.45%
        Final : 55.6% : v10 PRODUCTION MODEL
```

---

**Key Milestones:**
- **v1 (35%)**: Baseline with severe overfitting
- **v3 (42%)**: Fixed LR schedule breakthrough (+7%)
- **v8 (50%)**: ResNet architecture breakthrough (+8%)
- **v10 (55.6%)**: Enhanced augmentation - BEST MODEL (+5.6%)
- **v13-v14**: Failed experiments confirmed v10 is optimal

**What the chart shows:**
- 6 successful improvements (upward spikes at v3, v8, v10)
- 8 failed experiments (downward: v2, v4, v5, v9, v11, v13; flat: v6, v7, v12, v14)
- Systematic iteration: 35% → 55.6% (+20.6% total gain)
- v10 remains the production model despite 4 more attempts

---

## Key Architecture Decisions

| Component | v1 (Baseline) | v10 (Best) | Why Changed |
|-----------|---------------|------------|-------------|
| **CNN** | VGG-style | ResNet blocks | Skip connections for gradient flow |
| **LSTM Hidden** | 256 | 384 | Better capacity without overfitting |
| **Dropout** | None | 0.15-0.3 | Combat overfitting |
| **Augmentation** | Basic | Enhanced (shear, lines) | Match CAPTCHA distortions |
| **LR Schedule** | Fixed | Warmup + Plateau | Stable CTC alignment |
| **Loss** | CTC | CTC + Label Smoothing | Prevent overconfidence |
| **Optimizer** | Adam | AdamW + Weight Decay | Better regularization |

---

## How to Use This Model

```mermaid
sequenceDiagram
    participant User
    participant Preprocessor
    participant Model
    participant Decoder
    
    User->>Preprocessor: CAPTCHA Image (any size)
    Preprocessor->>Preprocessor: Remove black lines
    Preprocessor->>Preprocessor: Convert grayscale
    Preprocessor->>Preprocessor: Resize to 80x280
    Preprocessor->>Preprocessor: Normalize
    Preprocessor->>Model: Tensor (1, 1, 80, 280)
    
    Model->>Model: CNN forward
    Model->>Model: Reshape to sequence
    Model->>Model: BiLSTM forward
    Model->>Model: Classify each timestep
    Model->>Decoder: Log probs (70, 1, 63)
    
    Decoder->>Decoder: Greedy decode
    Decoder->>Decoder: Collapse repeats
    Decoder->>Decoder: Remove blanks
    Decoder->>User: Predicted text (e.g., "q02a9jk")
```

---

## Error Analysis: Length & Position

**Length Correctness** (shows CTC alignment quality):
```mermaid
pie title Length Correctness
    "Correct length (90.1%)" : 90.1
    "Too short (8.5%)" : 8.5
    "Too long (1.4%)" : 1.4
```

**Error Position Distribution** (no positional bias):
```mermaid
pie title Error Position Distribution
    "Start (0-1)" : 32.8
    "Middle (2-n)" : 37.7
    "End (last 2)" : 29.5
```

---

## Top Confusion Pairs (Directional counts)

Exact counts from v10 error analysis (directional):

| From | To | Count |
|------|----|-------|
| o | 0 | 15 |
| 0 | o | 14 |
| i | 1 | 14 |
| 5 | s | 9 |
| l | 1 | 7 |
| l | i | 7 |
| 1 | i | 5 |
| s | 5 | 5 |
| 1 | l | 5 |
| 8 | b | 4 |

| Pair | Total |
|------|-------|
| 0 ↔ o | 29 |
| i ↔ 1 | 19 |
| l ↔ 1 | 12 |
| 5 ↔ s | 14 |
| 8 ↔ b | 4 |

---

## Top Confusion Pairs (Bar Chart, Poster)

```mermaid
%%{init: {'theme':'base', 'themeVariables': {'xyChart': {'plotColorPalette': '#ff6b6b'}}}}%%
xychart-beta
    title "Top 5 Character Confusion Pairs"
    x-axis ["0 ↔ o", "i ↔ 1", "5 ↔ s", "l ↔ 1", "8 ↔ b"]
    y-axis "Error Count" 0 --> 30
    bar [29, 19, 14, 12, 4]
```

## Results & Metrics 

| Metric | Value |
|--------|-------|
| Sequence Accuracy (best) | 55.6% (epoch ~55–60) |
| Character Accuracy | 85.82% |
| Train Loss | 1.21 |
| Validation Loss | 1.39 |

---