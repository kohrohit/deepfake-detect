# Phase 2 — the pre-registered verdict

- **1_worst_fold_moved** [yes] paired delta on the baseline's worst LOGO fold is positive and its 95% interval clears zero
      {"delta": 0.2608448147052911, "delta_ci95": [0.24794992950473688, 0.2732782880174216], "fold": "sbi", "frozen": 0.7380272409392921, "tuned": 0.9988720556445831}
- **2_outside_its_null** [yes] the tuned run sits outside its own permutation null
      {"null": [0.2927771747055297, 0.7197104713144634], "permutation_p": 0.07692307692307693, "worst_fold_auc": 0.9949672720791951}
- **3_no_shortcut** [yes] the colour and JPEG controls stay flat
      {"colour": 0.5301742261648733, "colour_ceiling": 0.5800000000000001, "frozen_colour": 0.53, "jpeg": 0.5, "jpeg_structural": true}
- **4_clears_the_floor** [yes] capture-corpus session AUC >= 0.75
      {"floor": 0.75, "floor_inside_interval": true, "session_auc": 0.7593926917138446, "session_auc_ci95": [0.671613642428045, 0.8392337662337662], "session_tpr_at_zero_fpr": 0.13432835820895522}
- **5_capture_corpus_unbiased** [yes] capture-corpus session AUC, tuned against frozen (UNBIASED: the encoder never saw this corpus or its generator)
      {"delta": 0.09624292331446216, "frozen": 0.6631497683993824, "frozen_ci95": [0.5568763760686045, 0.762725462037962], "note": "The marginal intervals are reported but are NOT the test: both arms score the same sessions, so overlapping marginal intervals say nothing about a paired difference either way. `paired_ci95` is the test.", "paired_ci95": [-0.007179763739085726, 0.19926563721130777], "paired_delta": 0.09624292331446216, "paired_established": false, "paired_n_sessions": 125, "tuned": 0.7593926917138446, "tuned_ci95": [0.671613642428045, 0.8392337662337662]}

> **Question 1 is an upper bound, not a transfer number.** The tuned encoder was tuned on all five generator families, so a leave-one-generator-out fold is disjoint at the head and contaminated at the encoder. Question 5 is the unbiased comparison. If they disagree, believe 5.

**Tuned encoder ships: YES**
**Evidence floor cleared: YES** — **on the point estimate only. The 95% interval CONTAINS the floor**, so the measurement does not support the claim that slot D clears it. Question 4 was registered as a point-estimate test, which was a mistake; the question is left as registered and this is the correction.

**The unbiased improvement is NOT established**: paired delta +0.0962, 95% CI [-0.0072, +0.1993] over 125 sessions — the interval contains zero. The direction is positive on every honest measure and this corpus cannot resolve it. That is the finding, not a failure to finish.

Questions 1, 2 and 3 decide whether the tuned encoder replaces the frozen one. Question 4 is separate and decides nothing about shipping: it reports whether slot D clears the 0.75 evidence floor, which is necessary and not sufficient for the gate — dfd/service/evidence.py still applies the card's cross-corpus checks. A tuned worst fold above 0.75 does NOT open the gate.

Phase 1's weights stay shipped and the result is recorded as measured. That is the plan's own rule and a near miss does not renegotiate it.
