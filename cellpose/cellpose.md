# Laborbuch - cellpose

## Out of the box
- works and detects >80% but fails especially for large cells (by visual inspection)

## Grid search of diameter
(failed idea: multiple diameters + postprocessing to collect as many masks as possible)
- automatic detected diameter = 70 is best and has best sensitivity
- larger cells don't get detected better with larger diameter
- but smaller get merged

→ can't improve without retraining
→ or maybe normalization, though that is probably included?

---

go get data
