# 🐕 Greyhound Racing Modeling: Data Schema & Learning Architecture

We present a structured object model and modular learning pipeline for forecasting outcomes in greyhound racing. The system is designed to predict **win/place odds distributions** for each dog, based on rich structured metadata including weather, pedigree, standardized behavioral commentary, and market signals.

To ensure realistic forecasting, we model races **chronologically**, maintaining **per-dog memory vectors** instead of recursive histories. Our predictions are not deterministic ranks but **continuous odds distributions**, enabling both uncertainty modeling and market calibration.

---

Here is the **modified section of the README**, incorporating the newly implemented `Track` class and the new grouped-saving framework for `Dog` and `RaceParticipation` objects:

---

## 🧱 Core Data Classes

### ✅ 🐕 `Dog` class

Defines a greyhound and optionally maintains dynamic race data.

**✅ Implemented Fields:**

* `id` (formerly `dog_id`) — unique dog identifier
* `participations` — list of `RaceParticipation` objects
* Supports `add_participations()` to bulk-add races
* Can be serialized/deserialized via `pickle`
* Construction pipeline available in `build_and_save_dogs.py`
* **🔄 Dogs are now saved in grouped `.pkl` files based on ID hash buckets** (e.g. 400–499, 500–599)

**🔜 Planned:**

* Static fields: `name`, `birth_date`, `color`, `weight`, `trainer`
* Lineage: references to `sire`, `dam` (`Dog` instances or IDs)
* `memory_vector`: a learnable embedding to capture dynamic race form

---

### ✅ 🎟️ `RaceParticipation` class

Represents a **dog’s entry in a single race**.

**✅ Implemented Fields:**

* `race_id`, `trap_number`, `finish_time`, `position`, `odds`, etc.
* Structured `commentary_tags` as list of strings
* Built from raw scraped CSV rows via `parse_race_participation()`
* **Serialized using bucketed `.pkl` files based on race ID mod hash**

**✅ Status:**

* Used in `Dog`'s `participations`
* Core parsing logic complete

---

### ✅ 🏟 `Track` class

Defines the track a race was held on.

**✅ Implemented Fields:**

* `name`: string name of the track
* `race_ids`: list of race IDs run at this track
* `average_times`: dict mapping distances (in meters) to average finish times
* `race_count`: total number of races seen
* **Built dynamically from race participations** using `Track.from_race_participations()`
* **Serialized to individual files per track name**

**🔜 Planned Extensions:**

* `location`, `surface_type`, `geometry`
* Track bias features: drainage, sand composition

Tracks are cached and saved during test parsing or batch preprocessing.

---

### 🔜 🏁 `Race` class

Encapsulates a full race event, across **all dogs**.

**🔜 To Implement:**

* Fields:

  * `race_date`, `race_time`, `race_class`, `distance`, `category`, `prizes`
  * `rainfall_7d`: `[float; 7]`
  * `humidity`, `temperature`
  * `odds_vec`, `race_time_vec`, `commentary_tags_vec`

**Planned References:**

* List of participating `Dog` objects (including `trap_number`)
* `Track` object

This will be constructed later using `race_to_dog_index.pkl` and the `Dog` participations.

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


