# gtmhub_timeseries
This respository is created inorder to implement nbeats algorithm  [N-beats research paper](https://arxiv.org/pdf/1905.10437.pdf) with RevIN normalisation [RevIN paper](https://openreview.net/pdf?id=cGDAkQo1C0p).

### N-BEATS ALGORITHM:
> Nbeats architecture design methodology relies on a few key principles. First, the base architecture
should be simple and generic, yet expressive (deep). Second, the architecture should not rely on timeseries-specific feature engineering or input scaling. These prerequisites let us explore the potential
of pure DL architecture in TS forecasting. Finally, as a prerequisite to explore interpretability, the
architecture should be extendable towards making its outputs human interpretable. We now discuss
how those principles converge to the proposed architecture.

### RevIN:
> In this technique a new reversible instance normalization technique is introduced to alleviate the distribution shift problem in
time-series, which is known to cause a substantial discrepancy between the training and test data
distributions

### Model architecture:
>[Nbeats Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/nbeats.png)

>[Nbeats+RevIN Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/nbeats-revin.png)
