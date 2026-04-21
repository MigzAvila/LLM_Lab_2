# Exploratory Data Analysis (EDA) Playbook — Yelp Review Prediction

EDA is the discipline of *looking at the data before modelling it*. For the
Week5Lab1 crew we are predicting a user's `stars` and `review` text for a
target business, so our EDA is focused on **rating calibration** and
**theme extraction**, not general statistics.

This knowledge source tells every agent in the crew what "EDA anchors" to
compute and how to use them.

---

## 1. Why EDA matters here

A review prediction model fails in three predictable ways:

1. **Miscalibration** — the prose says "I loved it" but `stars = 2.0`, or
   vice versa.
2. **Prior drift** — the model ignores that this specific user is either a
   harsh rater (`average_stars ≈ 3.2`) or a generous one
   (`average_stars ≈ 4.7`).
3. **Hallucinated specifics** — the review mentions dishes, locations, or
   attributes that never appear in the retrieved evidence.

EDA catches (1) and (2) before the prediction step commits to a number.

---

## 2. The EDA probes every run should produce

For `user_id = {user_id}` and `item_id = {item_id}` the `eda_researcher`
agent should produce a compact bullet list with **six anchors**:

### User-side anchors

- **User average_stars** — from `search_user_profile_data`.
- **Retrieved per-review stars distribution** — from
  `search_historical_reviews_data` filtered to this user. Note the min,
  median, max and whether the distribution is:
  - *Consistent* (all within ±1 of the average), or
  - *Polarized* (bimodal: mostly 5s and 1s).
- **Tone fingerprint** — 3-5 high-signal adjectives or phrases this user
  reuses across reviews (e.g. "solid", "disappointing", "authentic").

### Item-side anchors

- **Item aggregate stars** — from `search_restaurant_feature_data`.
- **Retrieved per-review stars distribution** for this business — same
  tool probe as above but scoped to the item.
- **Theme split** — the 2-3 dominant *positive* themes and 1-2 dominant
  *negative* themes from retrieved reviews.

### Cross anchor (optional but recommended)

- **Prior drift flag** — if the user's `average_stars` is more than 0.8
  away from the item's aggregate stars, state the direction and say which
  prior the prediction should lean toward.

---

## 3. Calibration rules the prediction step must respect

These are the heuristics the `prediction_modeler` (and the `editor`) use
when translating anchors into a `stars` float.

1. **Anchor to the user, not the crowd.** Default the prediction to the
   user's `average_stars`, then adjust up or down based on whether the
   item's themes match the user's known preferences.
2. **Bound the adjustment.** A single adjustment step is usually no larger
   than ±1.0 around the user's prior. Exceed this only when the evidence
   is overwhelming (multiple retrieved reviews by this user on similar
   businesses at that extreme).
3. **Match prose to stars.**
   - `stars <= 2.0` → prose names at least one concrete complaint from the
     retrieved evidence.
   - `stars in [2.5, 3.5]` → prose is mixed; mention one positive and one
     negative theme.
   - `stars >= 4.0` → prose is net-positive but stays specific (dish, vibe,
     service) and avoids generic praise templates.
4. **Hedge when anchors are sparse.** If fewer than 2 retrieved reviews
   support the inference, stay within ±0.5 of the user's `average_stars`
   and make the prose cautiously specific.

---

## 4. Anti-patterns the crew must avoid

- Using the held-out target row (from `data/test_review_subset.json`) as a
  source of evidence. The test subset is **out of distribution** for the
  agents — never reference it.
- Emitting a review that names dishes, servers, or promotions that are not
  supported by the retrieved reviews.
- Letting the item's aggregate stars override the user's prior (a 4.5-star
  business still gets a 2.5 prediction from a user whose
  `average_stars = 2.7` and whose retrieved reviews are consistently harsh).
- Producing anything other than a single valid JSON object as the final
  artifact.

---

## 5. Output contract (shared with tasks.yaml)

The final crew output is always a single JSON object:

```json
{"stars": 4.5, "review": "The review text..."}
```

- `stars` is a float in `[1.0, 5.0]` (half-stars allowed).
- `review` is a first-person string (2-6 sentences) that sounds like the
  target user and is consistent with the EDA anchors above.
- No markdown, no preamble, no trailing commentary.
