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
>[Nbeats Only Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/images/architecture/nbeats.png) | [Nbeats+RevIN Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/blob/main/images/architecture/nbeats-revin.png)


### Benchmarks
##### *Benchmarking done on first 6 datasets*
[Nbeats+RevIN plots](https://github.com/yes-its-shivam/gtmhub_timeseries/tree/main/images/nbeats%2Brevin) | [Nbeats Only Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/tree/main/images/nbeats) | [Nbeats Only Architecture](https://github.com/yes-its-shivam/gtmhub_timeseries/tree/main/images/prophet) 
<table>
<tr><th>NBEATS+REVIN</th><th>NBEATS</th><th>PROPHET</th></tr>
<tr><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 | 0.00959	|0.000151	|0.012292|	0.020013 |
| 2 | 0.010154|	0.000156|	0.012475|	0.035633 |
| 3 | 0.055815|	0.067119|	0.259073|	0.127757 |
| 4 | 0.0273482|	0.1978104|	0.447588|	0.79791 |
| 5 | 0|	0|	0|	0 |
| 6 | 0.015047|	0.000423|	0.020556|	0.359561 |
  
 </td><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 |0.015679026|0.00040932448|0.020231768|0.032722607|
| 2 |0.010508158|0.00016971033|0.013027292|0.03687593|
| 3 |0.05063609|0.05599884|0.23664074|0.115894966|
| 4 |25.655844|1695.0132|41.170536|0.7474986|
| 5 | 0|0|0|0|
| 6 |0.015393304|0.0004451943|0.021099627|0.36776572|

</td><td>

| SNo. | MAE | MSE | RMSE | MAPE |
| --- | --- | --- | --- | --- |
| 1 | 0.08360422 | 0.028019099 | 0.16738907 | 0.17512284 |
| 2 | 0.45510322 | 1.0613955 | 1.0302405 | 1.5963151 |
| 3 | 4.924484 | 127.84147 | 11.3067 | 11.189242 |
| 4 | 51.486607|8275.656|90.970634|1.5083215|
| 5 | 0|0|0|0|
| 6 | 2.4326794|21.394512|4.62542|55.74821 |

</td></tr> </table>

### QUICK START

#### Dependencies
```
!git clone https://github.com/yes-its-shivam/gtmhub_timeseries.git
%cd gtmhub_timeseries
!pip install -r requirements.txt
```
##### In the nbeats-revin-forecast.py define the location to your .csv file, plot directory path, score file/dir path
```
df = pd.read_csv(YOUR FILE PATH HERE, parse_dates = ['timestamp'], index_col = 'timestamp') <-----
# sort by dates
df.sort_index(inplace = True)
df.drop('Unnamed: 0',axis=1,inplace=True)
.
.
.
plt.legend(['Real value train','Real value test','Prediction'])
plt.grid(True)
plt.savefig('FILENAME_PATH'+'.png') <-----

score=[]
for i,j in nbeats_revin_model_results.items():
  score.append(i+':'+str(j))
score_dict[str(filename)]= score

with open('PATH_TO_SCORE_FILE', 'w') as score_file: <-----
  score_file.write(json.dumps(score_dict))

print(str(filename)+' '+'Done!!!')
```

### Conclusions:

* The combined model is clearly working better then nbeats only architecture and facebook's prophet algorithm.
* The combined model is performing better for dataset with less amount of randomness
   * for dataset with more randomness either we need more data or we can try implementing nbeats with ensembling combined with RevIN
* After observing the graphs of forecast we can derive various insights for business usecases
