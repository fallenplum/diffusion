# 预锁规则下的初步审计判断

|comparison|scope|decision|both_metrics_positive|all3_seeds_both_positive|corrected_both_intervals_positive|budget_sensitive|
|---|---|---|---|---|---|---|
|Real128-Null128|primary|DIRECTIONAL_ONLY|True|True|False|False|
|Real128-Wrong128|primary|DIRECTIONAL_ONLY|True|True|False|False|
|Real512-Null512|primary|DIRECTIONAL_ONLY|True|False|False|False|
|Real512-Wrong512|primary|INCONCLUSIVE|False|False|False|False|
|DID512vs128|primary|INCONCLUSIVE|False|False|False|False|
|Real128-A1|primary|DIRECTIONAL_ONLY|True|True|False|False|
|Real1024-Null1024|secondary|DIRECTIONAL_ONLY|True|True|False|False|
|Real1024-Wrong1024|secondary|INCONCLUSIVE|False|False|False|False|
|DID1024vs128|secondary|INCONCLUSIVE|False|False|False|False|
|Real512-A1|secondary|DIRECTIONAL_ONLY|True|True|False|False|
|Real1024-A1|secondary|DIRECTIONAL_ONLY|True|False|False|False|
|OriginalHallmark-Group0|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedTrue-Random128|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedTrue-Random512|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedRandom-MatchedSet128|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedRandom-MatchedSet512|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedTrue-Original128|secondary|INCONCLUSIVE|False|False|False|False|
|EnrichedTrue-Original512|secondary|INCONCLUSIVE|False|False|False|False|
|NoAnchorReal-Null|secondary|INCONCLUSIVE|False|False|False|False|
|NoAnchor-Anchored|secondary|INCONCLUSIVE|False|False|False|False|
|MeanOtherGroups-Group0|secondary|INCONCLUSIVE|False|False|False|False|
|Mean8Groups-Null|secondary|DIRECTIONAL_ONLY|True|True|False|False|
|Mean8Groups-A1|secondary|DIRECTIONAL_ONLY|True|True|False|False|

这是第一轮可复算证据，不构成外部审计或临床泛化裁判。
