# 🐕 Greyhound Racing Modeling: Data Schema & Learning Architecture

We present a structured object model and modular learning pipeline for forecasting outcomes in greyhound racing. The system is designed to predict **win/place odds distributions** for each dog, based on rich structured metadata including weather, pedigree, standardized behavioral commentary, and market signals.

To ensure realistic forecasting, we model races **chronologically**, maintaining **per-dog memory vectors** instead of recursive histories. Our predictions are not deterministic ranks but **continuous odds distributions**, enabling both uncertainty modeling and market calibration.

---

## 🧱 Core Data Classes

### 🏁 `Race` class

Encapsulates a single race event.

**Fields:**

* `race_date`, `race_time` — when the race occurred.
* `race_class`, `distance`, `category`, `prizes`.
* **Weather conditions**:

  * `rainfall_7d`: `[float; 7]` — rainfall (mm) per day before the race.
  * `humidity`: Float
  * `temperature`: Float (°C)
* **Dog race-level data**:

  * `odds_vec`: `[float; num_dogs * 2]` — win and place odds before the race for each dog.
  * `race_time_vec`: `[float; num_traps]` — actual finish time per trap.
  * `commentary_tags_vec`: `[list[str]; num_dogs]` — standardized tags describing behavior (e.g., `SAw`, `Ld1/2`, `EvCh`).

**References:**

* List of participating `Dog` objects (including `trap_number`, `dog_id`, etc.)
* Associated `Track` object

---

### 🐕 `Dog` class

Defines a greyhound and optionally maintains a dynamic memory.

**Static fields:**

* `dog_id`, `name`, `birth_date`, `color`, `weight`, `trainer`
* Lineage: references to `sire` and `dam` (each another `Dog` object)

**Dynamic fields:**

* `memory_vector`: a learned embedding of performance form (updated chronologically)
* No recursive storage of race history — it’s modeled via `memory_vector` during training.

---

### 🏟 `Track` class

Represents track properties and identity.

* `name`, `location`, `surface_type`, `geometry`
* May include drainage features, sand type, or baseline condition indicators

Each race references a track to inform how local conditions (combined with weather) may impact performance.

---

## 📉 Target: Predicting Odds Distributions

We model each dog's predicted odds as **continuous probability distributions** for:

* **Win Odds**: domain $(1, \infty)$
* **Place Odds**: domain $(1, \infty)$

### 🎯 Why Distributions?

* Real-world markets express confidence through odds.
* Predicting point estimates or ranks loses this information.
* Modeling a **distribution over odds** captures both:

  * **Sharp market confidence** (e.g. a strong favorite winning)
  * **Upsets** (dog with long odds unexpectedly placing)

---

## 🔬 Constructing Target Distributions

From:

* Final **market odds** just before the race,
* Actual **finishing times** (from `race_time_vec`),

We construct a target distribution that:

* Is **tall and narrow** if the result aligns with market expectation (e.g., low odds dog wins).
* Is **short and wide** if an unexpected result occurs (e.g., long odds dog places or wins).

### 📈 Suggested Distribution Family:

We propose using the **Translated Gamma Distribution**:

* Defined on $(1, \infty)$, modeling odds directly.
* Flexible: accommodates skew, scale, and location shifts.
* Can be parameterized to match market shape and real result.

We may also evaluate:

* Log-Normal
* Scaled Beta Prime
* Generalized Pareto

These options provide **robust modeling of heavy tails** for rare upsets, and narrow peaks for confident wins.

Loss functions (e.g. KL divergence or Wasserstein distance) can compare predicted distributions to targets.

---

## 🧠 Commentary Encoding: Structured Tags

Each dog's race is annotated with standardized **commentary tags**, not raw text:

* Examples:

  * `SAw` – Slowly Away
  * `Ld1/2` – Led halfway
  * `EvCh` – Every Chance
* Stored as `commentary_tags_vec` (list per dog) in the `Race` class.
* Encoded using tag embeddings or one-hot + dense layers.

This provides behavioral context (e.g., break speed, lane preference) in a machine-readable format.

---

## 🔁 Chronological Training with Per-Dog Memory

Rather than recursively embedding full race histories, we train on **chronologically sorted races**. Each dog maintains a learned **memory vector** representing latent form (like an RNN hidden state).

### Training Loop:

```plaintext
for race in all_races_sorted_by_date:
    for dog in race.dogs:
        h = get_memory(dog.id) or h0
        x = build_features(race, dog, h)

    y_pred = model(x)
    loss = compare_distributions(y_pred, target_distribution)
    backpropagate()

    for dog in race.dogs:
        h_new = memory_update_model(h, race_result_features)
        set_memory(dog.id, h_new)
```

### 🧠 Memory Update Module

* Learns how to update a dog’s state vector from each race.
* Could be:

  * A GRU cell
  * Simple MLP
  * ELO-style Bayesian update

This enables a **scalable and causally valid** representation of dynamic dog form.

---

## 🧱 Full Race Object Schema (Markdown Diagram)

```markdown
Race {
  race_date: datetime
  race_time: datetime
  distance: int
  race_class: str
  category: str
  rainfall_7d: [float; 7]
  humidity: float
  temperature: float
  odds_vec: [float; num_dogs * 2]        # win/place odds
  race_time_vec: [float; num_traps]      # actual finish times
  commentary_tags_vec: [list[str]]       # structured tags per dog

  Track: {
    name: str
    location: str
    surface_type: str
  }

  dog_list: [
    {
      id: str
      birth_date: datetime
      trainer: str
      weight: float
      color: str
      sire: Dog
      dam: Dog
      memory_vector: [float]
    },
    ...
  ]
}
```

---

## 🧩 Model Components Summary

| Feature                        | Method                               |
| ------------------------------ | ------------------------------------ |
| **Pedigree**                   | GNN on sire/dam tree                 |
| **Form (past races)**          | Per-dog memory vector                |
| **Rainfall / Humidity / Temp** | Dense layers                         |
| **Commentary Tags**            | Embedding table                      |
| **Track ID, Category**         | Embedding layers                     |
| **Odds Prediction**            | Translated Gamma / Log-Normal / etc. |

---

## ✅ Benefits

* Full causal and temporal validity (no lookahead)
* Robust to market mispricings
* Scalable: no recursion or full history tracking
* Reflects domain knowledge (e.g. behavioral patterns, weather)
* Provides **uncertainty-aware** odds forecasting


