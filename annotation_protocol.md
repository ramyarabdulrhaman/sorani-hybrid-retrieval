# Protocol for a future independent retrieval benchmark

This is a protocol, not completed annotation. Three Sorani annotators were
reported as available, but their qualifications have not been documented. No
recruitment records, consent forms, newly judged examples, verified source-
rights documents, or human dialect labels are present in the workspace.

1. Select questions and passages from a dataset with documented permission for
   the intended research and redistribution. Preserve source, version, licence,
   attribution, and evidence of the underlying text owners' permission. Verify
   the dataset terms and provenance before redistribution.
2. Obtain genuinely new, held-out questions and passages not used during method
   design, model training, validation or previous experiments. Check exact and
   near-duplicate overlap with prior data and report limitations of these checks
   and possible pretrained-model exposure. Freeze splits and the analysis plan
   before inspecting test results.
3. Use two independent qualified Sorani annotators per sampled item. Use the
   third annotator to adjudicate disagreements; this three-person arrangement
   deviates from the upgrade plan's preferred four-to-six-person panel. Record
   qualifications, instructions, assignments, consent, compensation, and
   conflicts of interest. Label passage variety independently (Sorani, other,
   mixed, or uncertain); do not infer dialect from automatic heuristics alone.
4. Pool candidates for each question from multiple lexical, dense and fusion
   systems, including systems not under evaluation where feasible. Blind system
   names and randomize presentation order. Permit multiple relevant passages.
   Record each question ID, passage ID, annotator ID, relevance (relevant,
   not relevant, or uncertain), dialect, and a short justification. Keep
   uncertain distinct from negative. Include the linked passage in each pool.
5. Measure and report inter-annotator agreement, disagreements, missing
   judgments and the number of relevant passages per query. Have an independent
   adjudicator resolve disagreements using the original question and evidence.
   Archive original judgments and adjudicated labels separately.
6. Publish only material permitted by its rights holders. Release the protocol,
   split definitions, judgments or permissible identifiers, pool construction,
   scripts, and model settings. Clearly flag unjudged passages and report pool
   coverage and sensitivity to incomplete judgments. Evaluate Recall@k,
   MRR@10 and nDCG using multiple judged positives, paired query- or
   appropriate cluster-level uncertainty, and prespecified comparisons.
