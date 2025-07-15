# Data Model for Greyhound Racing

We propose three main classes – **Race**, **Dog**, and **Track** – to capture the rich, structured data of greyhound racing. Each **Race** object holds its basic metadata (date, distance, class, prizes, etc.) and references to participating dogs and the track.  In particular, we include a `rainfall_7d` attribute (a 7-element vector of recent rainfall) because track conditions are highly weather‐dependent – official track guidelines note that **“weather is the main external factor”** affecting track preparation (with rainfall explicitly affecting how wet or “sloppy” the sand becomes).  Empirical analysis of horse races likewise shows that persistent rain (turning turf soft/yielding) forces horses to expend extra energy and yields slower race times.  By analogy, a greyhound race class should store the amount of rain over the past week to let a model account for how a muddy or dry surface might alter performance.

* **Race class:** include fields like `race_date`, `race_time`, `race_class`, `distance`, `prizes`, etc.  Also include a reference to a **Track** object and a list of the participating **Dog** objects (with their trap numbers and results).  Crucially, add a `rainfall_7d` array (e.g. millimeters of rain each day for the 7 days before the race) so the model can learn how recent weather affected that race.

* **Track class:** include the track’s identity (name, location or region) and static properties (surface type/geometry).  Real-world track maintenance manuals emphasize monitoring moisture at each track, e.g. using probes and weather stations to keep the surface consistent.  For example, overly wet (“sloppy”) or overly dry conditions are known to alter greyhound exertion and injury risk.  The Track class could store typical baseline conditions (like optimal water content or sand type) and facilities (e.g. drainage, covers).  It should be linked from each Race so that race-specific preparation (given the same Track) can be combined with weather history.

* **Dog class:** store each individual greyhound’s static attributes and dynamic history.  Static attributes include `dog_id`, `name`, trainer, birth date, weight, color, etc.  Crucially, for performance modeling we attach the dog’s past race results: an *ordered list* of prior Race entries (each with date, position, time, etc.) – this gives the dog’s historical form.  (We expect dogs to run many races; for example, databases often show dozens of races per dog – one greyhound’s record was listed as “55 races” – and while very few exceed \~150, our class should allow arbitrarily long race histories.)  The Dog class should also capture **lineage**: references to its `sire` and `dam` (each of which can be another Dog object), forming a pedigree tree.  Greyhound pedigree databases routinely trace at least 5–6 generations of ancestry. We will store each dog’s full ancestry tree (as nested objects or IDs) but only include the immediate parent references in each Dog instance – for example, the dam’s own pedigree is stored in the dam’s object, not duplicated in the child.

With this structure, each Dog object implicitly embeds a pedigree **tree**, and each Race links together many Dog objects.  For model input, we will **flatten** these structures into fixed‐size features.  One approach is to compute a performance rating for each dog from its race history – analogous to Elo ratings in chess.  In fact, racing analytics often use Elo or similar methods to aggregate career performance: an Elo-based index can capture a horse’s entire career performance across wins, placings, and earnings.  Likewise, we could compute an **Elo rating** (or other ranking score) from each greyhound’s past finishes to summarize its ability.

Alternatively (or additionally), we might convert each dog’s race history into feature vectors: for example, summary stats (wins, average speed, etc.) or a sequential embedding of the ordered race outcomes.  Pedigree data can similarly be encoded.  In genetics and risk prediction, pedigrees are often represented as directed graphs (parents→offspring) and then flattened to fixed-size matrices for neural nets.  For instance, one method builds a matrix \$H\$ whose rows are individuals (ancestors) with columns for features and pointers to their parent indices.  We can adopt similar encoding: e.g. enumerating all known ancestors up to a cutoff generation and including each ancestor’s attributes, or applying graph/sequence embedding techniques to the pedigree.  The key is to turn the variable‐size dog-history and family‐tree into a consistent input.

**Performance/Rating Calculation:**  As a concrete example, an *Elo-like rating* can be derived from the chronological race results of each dog.  This captures how a dog performed *relative to its competition over time*.  Research in equine sports notes that an Elo-based index “allows evaluation of a horse by considering various different traits such as wins, placings, earnings over the entire career altogether”.  In practice, we would run through each past race, updating the dog’s rating based on finishing order (perhaps weighted by class/distance), yielding a single score per dog for model training.  This is just one way to flatten the time-series performance data; other detailed representations (like sequences or summary stats of the last N races) could preserve more nuance if needed.

**Summary:**

These classes capture all necessary details: race metadata, weather (rainfall vector), dog pedigrees, and performance histories.  When preparing data for a learning model, we can flatten each Dog’s history into features (e.g. computing a career Elo score or encoding sequences of past finishes), and similarly encode pedigree trees in a fixed-size form (e.g. a matrix of ancestor features). This rich, object‐oriented schema (with citations above) ensures no critical detail is lost while still yielding a consistent input format for modeling.

## Suggested Model Architecture Overview

To learn from this structured and recursive data, we propose a **hybrid model** that combines modern deep learning components tailored to each substructure (pedigree, race history, text commentary, etc.). This modular design allows us to preserve the detail-rich hierarchy of the Race-Dog-Track ecosystem, while still producing fixed-size representations suitable for training and inference.

### Key Components

* **Pedigree Embedding**:
  Each dog’s ancestry (referenced via `sire` and `dam`) forms a tree of interconnected Dog objects. We will represent this pedigree using a **Graph Neural Network (GNN)** — allowing us to embed recursive family trees into a vector that captures inherited traits from multi-generational lineage. The GNN input graph contains Dog nodes with features (birth date, weight, race record summary, etc.) and edges for parent-offspring relationships.

* **Race History Embedding**:
  Each Dog’s ordered list of past races (with outcomes, timings, conditions, etc.) will be flattened either via:

  * A sequential model (e.g. GRU or Transformer) to encode patterns over time,
  * Or by summary statistics (e.g. win %, best average time) if a more efficient fixed-length input is desired.
    Optionally, we may compute an **Elo-style rating** from past results to compactly represent performance.

* **Rainfall Vector (7 days)**:
  The `rainfall_7d` vector is passed through a simple dense layer, since its structure is consistent and small (7 elements). No temporal model is needed here, but the pattern (e.g. dry then heavy rain) can still be learned through the weights.

* **Commentary Encoding**:
  Dog-specific race comments (e.g. “MidTW, Ld 1/2”) are embedded using a pretrained language model such as **DistilBERT** or **Sentence-BERT**, producing contextual features for race behavior.

* **Track & Race Metadata**:
  Features like track ID, surface type, temperature, and race class/distance are processed by standard dense layers or embedding layers for categorical data.

All embeddings are concatenated to form a **per-dog feature vector**, which is passed through a fully connected head to predict **race outcomes** (win/place probabilities, rank likelihood, or margin of finish).

---

## Data Structure Visualization

The following shows the recursive, object-oriented layout of a single race. This is how the classes relate, and how each encapsulates deeper histories or structures:

```markdown
Race {
  race_times_vec: [float],               # Race time for each dog
  dog_commentary_vec: [str],            # Commentary per dog
  track_length: int,                    # e.g. 460 meters
  category: str,                        # e.g. A10
  Track: {
    location: str,
    date_time: datetime,
    rainfall_7d: [float; 7],           # Daily rainfall leading up to race
    temp: float,
    humidity: float
  },
  dog_dict: {
    dog1: {
      id: str,
      birth_date: datetime,
      weight: float,
      trainer: str,
      sire: dog1.1,                    # Parent dog (recursive link)
      dam:  dog1.2,
      past_races_list: [              # Dog's prior races
        Race1.1, Race1.2, ...
      ]
    },
    dog2: { ... },
    ...
    dog6: { ... }
  }
}
```

This layout is **recursive** and **graph-like**:

* Each dog’s parents are also dogs, potentially with their own parents.
* Each dog has raced before in other races, which include other dogs and tracks.
* All of this is linked, so we can extract graphs, sequences, and structured metadata for modeling.

---

## Why This Architecture?

This hybrid design lets us:

* Retain full race and lineage structure during training.
* Encode both short-term form (recent races) and long-term traits (lineage).
* Learn patterns in commentary text that correlate with tactical behaviors (e.g. crowding, overtaking).
* Avoid throwing away domain-specific signals like rainfall history and track identity.

In short: **we respect the recursive structure** of greyhound racing data while preparing it for efficient machine learning input.

---

Would you like to extend the README next with an example input JSON (after flattening), or a sample training pipeline sketch in PyTorch?


**Sources:** Greyhound racing database examples; horse racing analytics on rainfall and Elo ratings; GBGB track manual on weather and surface preparation; pedigree-modeling research.
