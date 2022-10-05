# gtmhub_timeseries
This respository is created inorder to implement nbeats algorithm introduced in [Neural Basis Expansion Analysis For
Interpretable Time Series](https://arxiv.org/pdf/1905.10437.pdf) with RevIN normalisation introduced in [Reversible Instance Normalisation For
Accurate Time-Series Forecasting Against
Distribution Shift](https://openreview.net/pdf?id=cGDAkQo1C0p).

## ABSTRACT

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

### Model Architectures:
>[Nbeats Only Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/nbeats.png) | [Nbeats+RevIN Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/nbeats-revin.png)


### Benchmarks

<table>
<tr><th>NBEATS ONLY </th><th>PROPHET</th><th>NBEATS+REVIN</th></tr>
<tr><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 |0.015679026|0.00040932448|0.020231768|0.032722607|
| 2 |0.010508158|0.00016971033|0.013027292|0.03687593|
| 3 |0.05063609|0.05599884|0.23664074|0.115894966|
| 4 |25.655844|1695.0132|41.170536|0.7474986|
| 5 | 0.0|0.0|0.0|0.0|
| 6 |0.015393304|0.0004451943|0.021099627|0.36776572|

</td><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 | 0.08360422 | 0.028019099 | 0.16738907 | 0.17512284 |
| 2 | 0.45510322 | 1.0613955 | 1.0302405 | 1.5963151 |
| 3 | 4.924484 | 127.84147 | 11.3067 | 11.189242 |
| 4 | 51.486607|8275.656|90.970634|1.5083215|
| 5 | 0.0|0.0|0.0|0.0|
| 6 | 2.4326794|21.394512|4.62542|55.74821 |
  
</td><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 |  |  |  | |
| 2 |  |  |  | |
| 3 |  |  |  | |
| 4 |  |  |  | |
| 5 |  |  |  | |
| 6 |  |  |  | |

</td></tr> </table>

### QUICK START

#### Dependencies
```
!git clone https://github.com/yes-its-shivam/gtmhub_timeseries.git
%cd gtmhub_timeseries
!pip install -r requirements.txt
```
##### In the nbeats-revin-forecast.py define the location to your .csv file
```
df = pd.read_csv(YOUR FILE PATH HERE, parse_dates = ['timestamp'], index_col = 'timestamp')
# sort by dates
df.sort_index(inplace = True)
df.drop('Unnamed: 0',axis=1,inplace=True)
```

### Conclusions:

* The combined model is performing better for dataset with less amount of randomness
   * for dataset with more randomness either we need more data or we can try implementing nbeats with ensembling combined with RevIN
* Overall performance of the combined model is better then just nbeats model
* Overall performance
      
