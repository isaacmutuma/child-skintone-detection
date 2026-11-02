# Child-Detection Skin-Tone Audit

Do automated child detectors, used to filter text-to-image training data for child safety, miss darker-skinned children more often?

This project extends Crețu et al., *Evaluating Concept Filtering Defenses against Child Sexual Abuse Material Generation by Text-to-Image Models* (IEEE S&P 2026, arXiv:2512.05707). That paper benchmarked child detectors but did not analyse performance by skin tone.

- **Spec:** [EXPERIMENT_PLAN.md](EXPERIMENT_PLAN.md)
- **Hypotheses (frozen before results):** [PREREGISTRATION.md](PREREGISTRATION.md)
- **Running it with an agent:** [AGENT_KICKOFF.md](AGENT_KICKOFF.md)

**Ethics:** no CSAM is generated or handled. Only public research face datasets and ICCWD are used. No images or per-image labels are committed. Figures contain charts only.
