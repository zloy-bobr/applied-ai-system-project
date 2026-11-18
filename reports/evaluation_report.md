# Evaluation report

Provider: `offline`  |  k = 5  |  15 labeled requests

## Ranking quality

| Metric | Agent | Popularity baseline |
|---|---|---|
| Hit@1 (top pick is relevant) | 1.00 | – |
| Precision@5 | 0.52 | – |
| Recall@5 | 0.95 | – |
| NDCG@5 | 0.94 | 0.21 |
| Avg confidence | 0.53 | – |
| Genre diversity | 0.88 | – |
| Artist diversity | 1.00 | – |

Precision@5 is capped by small relevant sets (e.g. only 1 r&b song exists), so Recall and NDCG are the fairer measures.

| Request | Top 3 | Hit@1 | Recall@5 | NDCG@5 | Conf. |
|---|---|---|---|---|---|
| upbeat happy pop for a sunny morning | Sunrise City, Rooftop Lights, Gym Hero | 1 | 1.00 | 0.97 | 0.68 |
| chill lofi beats to study to | Midnight Coding, Library Rain, Spacewalk Thoughts | 1 | 0.75 | 0.83 | 0.61 |
| something heavy and aggressive for the gym | Gym Hero, Storm Runner, Lightspeed Rave | 1 | 1.00 | 1.00 | 0.70 |
| sad acoustic songs for a rainy night | Paper Boats, Library Rain, Midnight Coding | 1 | 0.67 | 0.77 | 0.50 |
| dreamy synthwave for a late night drive | Night Drive Loop, Storm Runner, Velvet Promise | 1 | 1.00 | 0.85 | 0.33 |
| romantic slow r&b for date night | Velvet Promise, Midnight Coding, Paper Boats | 1 | 1.00 | 1.00 | 0.42 |
| relaxing jazz in a coffee shop | Coffee Shop Stories, Library Rain, Dusty Porch Swing | 1 | 1.00 | 1.00 | 0.44 |
| classical music to help me sleep | Moonlit Sonata Redux, Spacewalk Thoughts, Focus Flow | 1 | 1.00 | 1.00 | 0.49 |
| party music with lots of energy | Lightspeed Rave, Gym Hero, Sunrise City | 1 | 1.00 | 1.00 | 0.75 |
| spanish songs for the beach | Island Time, Rooftop Lights, Sunrise City | 1 | 1.00 | 1.00 | 0.65 |
| angry metal | Iron Cathedral, Storm Runner, Night Drive Loop | 1 | 1.00 | 1.00 | 0.48 |
| nostalgic country music | Dusty Porch Swing, Night Drive Loop, Paper Boats | 1 | 1.00 | 0.92 | 0.26 |
| confident hip hop with swagger | Golden Hour Bars, Gym Hero, Coffee Shop Stories | 1 | 1.00 | 1.00 | 0.23 |
| instrumental focus music with no vocals | Focus Flow, Library Rain, Spacewalk Thoughts | 1 | 0.80 | 0.87 | 0.64 |
| feel-good songs from the 2010s | Rooftop Lights, Sunrise City, Lightspeed Rave | 1 | 1.00 | 0.92 | 0.80 |

## Reliability and guardrails: 10/10 passed

| Case | Expected | Got | Pass |
|---|---|---|---|
| empty request | blocked | blocked | ✅ |
| too long | blocked | blocked | ✅ |
| gibberish | clarification | clarified | ✅ |
| off-topic | clarification | clarified | ✅ |
| prompt injection | warning | clarified | ✅ |
| injection + real request | recommendations | 5 recs | ✅ |
| genre not in catalog | recommendations | 5 recs | ✅ |
| language not in catalog | recommendations | 5 recs | ✅ |
| messy capitalization | recommendations | 5 recs | ✅ |
| consistency (same request twice) | identical | identical | ✅ |

## Bias check (across all labeled requests)

- Catalog coverage: 100% of songs were recommended at least once
- Never recommended: none
- Most recommended songs: Spacewalk Thoughts (7), Coffee Shop Stories (7), Gym Hero (6), Library Rain (6), Sunrise City (5)
- Top artists: Neon Echo (9), LoRoom (8), Orbit Bloom (7)
- Mean popularity of recommendations: 60.8 (catalog average 61.7)
- Genre exposure vs. catalog share (>1 = over-exposed): ambient 1.68, jazz 1.68, pop 1.32, folk 1.2, lofi 1.12, rock 0.96, synthwave 0.96, indie pop 0.96, edm 0.96, hip hop 0.72, classical 0.72, country 0.72, metal 0.48, r&b 0.48, reggae 0.48
