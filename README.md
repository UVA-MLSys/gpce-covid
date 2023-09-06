# Interpreting County-Level COVID-19 Infections using Transformer and Deep Learning Time Series Models

## Introduction
In this work, we utilize the Temporal Fusion Transformer (TFT) to achieve state-of-the-art performance in forecasting US county-level infections while enabling new forms of interpretability through analyzing complex spatiotemporal patterns. The proposed model (1) outperforms other popular deep learning models in all evaluation metrics for multivariate multi-horizon forecasting, (2) exhibits robust performance in predicting non-stationary trends of the infections at different waves of the COVID-19 pandemic, (3) interprets temporal patterns, such as weekly and holiday seasonality in reported cases, through multi-head attention weights, and (4) reveals spatial patterns using attention weights that are correlated to the infection spread. The model performs consistently across different counties, despite the large variation in infection rates, and can be easily extended to other datasets at the community level, as exemplified in the characteristics (e.g. population, health status, and socioeconomic factors).

## Folder Structure

* **Archives**: Unused codes.
* **dataset_raw**: Contains the collected raw dataset and the supporting files. To update use the [Update dynamic dataset](/dataset_raw/Update%20dynamic%20features.ipynb) notebook. The static dataset is already updated till the onset of COVID-19 using [Update static dataset](/dataset_raw/Update%20static%20features.ipynb) notebook.
* **papers**: Related papers. 
* **Related Works**: Contains the models and results used to compare the TFT performance with related works. 
* **TFT-PyTorch**: Contains all codes and merged feature files used during the TFT experimentation setup and interpretation. For more details, check the [README.md](/TFT-PyTorch/README.md) file inside it. The primary results are highlighted in [results.md](/TFT-PyTorch/results.md). 


## How to Reproduce

For detailed instructions on how to reproduce, follow the [Reproduce.md](/Reproduce.md) file. In summary, it includes the following steps:

* Getting the env ready
  * From scratch using Anaconda or pip and installing libraries using [requirements.txt](/requirements.txt).
  * Creating the containers (`Singularity` or `Docker`). Definitions are given in [singularity.def](/singularity.def) and [Dockerfile](/Dockerfile). An already created `Singularity` container is hosted [here](library://khairulislam/collection/tft_pytorch:latest).
* To reproduce tft experiments run the [scripts](/TFT-pytorch/script/) in the [TFT-pytorch](/TFT-pytorch) folder.
* For the related works comparison run the scripts from [Related Works](/Related%20Works/) folder.
* Note that, for Python path management, the scripts have to be run from their corresponding folder (not from this root).
* The are some notebooks available for most scripts too for easier debugging.

## Results
Results on all 3,142 US counties are listed below.

### Ground Truth
![](/TFT-pytorch/results/ground_truth.jpg)

### Benchmark

Test result comparison of TFT with five other deep learning models.
![](/TFT-pytorch/results/TFT_baseline/figures/Test_comparison.jpg)

### Temporal Patterns

Time series data typically exhibit various temporal patterns,
such as trend, seasonal, and cyclic patterns. Here we investigate how well our TFT model can learn and interpret these patterns by conducting
experiments on data with these patterns.

1. Attention weights aggregated by past time index showing high importance in the `same day the previous week` (position index -7). 

![Train_attention.jpg](/TFT-pytorch/results/TFT_baseline/figures/Train_attention.jpg)

2. Weekly `seasonality` due to reporting calculated using auto-correlation at different lag days $k \in [1, 21]$. Our analysis shows a clear weekly periodicity, where the correlation peaks at lag day k = 7.  This is attributed to the weekly reporting style from hospitals, leading to fewer reported cases on weekends.

![seasonal_pattern.jpg](/TFT-pytorch/results/TFT_baseline/figures/seasonal_pattern.jpg)

3. `Cyclic` holiday patterns (Thanksgiving, Christmas). During holidays, hospitals and COVID-19 test centers often have reduced staffing and operating hours, leading to fewer tests and reported cases. Leading to a drop in attention for those days. 

![holiday_pattern.jpg](/TFT-pytorch/results/TFT_baseline/figures/holiday_pattern.jpg)

4. `Trend`: TFT model's test performance on all US counties for additional data splits learning different infection trends. 

![Test_splits.jpg](/TFT-pytorch/results/TFT_split_3/figures/Test_splits.jpg)

### Spatial Patterns

Spatial distribution of COVID-19 cases in US counties and corresponding attention weights from TFT.

1.  Cumulative COVID-19 cases across US counties 
![](/TFT-pytorch/results/TFT_baseline/figures/maps/cases_quantiles.jpg)
2. Avg. attention weights across US counties from TFT 
![](/TFT-pytorch/results/TFT_baseline/figures/maps/attention_quantiles.jpg)

## Features

Note that, past values of the target and known futures are also used as observed inputs by TFT.

<div align="center">

<table border="1">
<caption> <h2>Details of Features </h2> </caption>
<thead style="border:2px solid">
<tr>
<th>Feature</th>
<th>Type</th>
<th>Update Frequency</th>
<th>Description/Rationale</th>
<th>Source(s)</th>
</tr>

</thead>
<tbody>
<tr>
<td><strong>Age Distribution</strong> <br> (% age 65 and over)</td>
<td rowspan="2">Static</td>
<td rowspan="2">Once</td>
<td><em>Aged 65 or Older from 2016-2020 American Community Survey (ACS)</em>. Older ages have been associated with more severe outcomes from COVID-19 infection.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td><strong>Health Disparities</strong> <br>(Uninsured)</td>
<td><em>Percentage uninsured in the total civilian noninstitutionalized population estimate, 2016- 2020 ACS</em>. Individuals without insurance are more likely to be undercounted in infection statistics, and may have more severe outcomes due to lack of treatment.</td>
<td><span><a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a></span></td>
</tr>

<tr>
<td><strong>Transmissible Cases</strong></td>
<td rowspan="4">Observed</td>
<td rowspan="7">Daily</td>
<td><em>Cases from the last 14 days per 100k population</em>. Because of the 14-day incubation period, the cases identified in that time period are the most likely to be transmissible. This metric is the number of such "contagious" individuals relative to the population, so a greater number indicates a more likely continued spread of disease.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a> , <a href="https://svi.cdc.gov/data-and-tools-download.html" target="_blank">2020 SVI</a> (for population estimate)</span></span></td>
</tr>

<tr>
<td><strong>Disease Spread</strong></td>
<td><em>Cases that are from the last 14 days (one incubation period) divided by cases from the last 28 days </em>. Because COVID-19 is thought to have an incubation period of about 14 days, only a sustained decline in new infections over 2 weeks is sufficient to signal a reduction in disease spread. This metric is always between 0 and 1, with values near 1 during the exponential growth phase, and declining linearly to zero over 14 days if there are no new infections.</td>
<td><span><span><a href="https://usafacts.org/issues/coronavirus/" target="_blank">USA Facts</a></span></span></td>
</tr>

<tr>
<td><strong>Social Distancing</strong></td>
<td><em>Unacast social distancing scoreboard grade is assigned by looking at the change in overall distance traveled and the change in nonessential visits relative to baseline (previous year), based on cell phone mobility data</em>. The grade is converted to a numerical score, with higher values being less social distancing (worse score) is expected to increase the spread of infection because more people are interacting with other.</td>
<td><span><a href="https://www.unacast.com/covid19/social-distancing-scoreboard" target="_blank">Unacast</a></span></td>
</tr>

<tr>
<td><strong>Vaccination Full Dose</strong><br>(Series_Complete_Pop_Pct)</td>
<td> Percent of people who are fully vaccinated (have a second dose of a two-dose vaccine or one dose of a single-dose vaccine) based on the jurisdiction and county where the recipient lives.</td>
<td><span><a href="https://data.cdc.gov/Vaccinations/COVID-19-Vaccinations-in-the-United-States-County/8xkx-amqh" target="_blank">CDC</a></span></td>
</tr>

<tr>
<td><strong>SinWeekly</strong></td>
<td rowspan="2">Known Future</td>
<td> <em>Sin (day of the week / 7) </em>.</td>
<td rowspan="2">Date</td>
</tr>

<tr>
<td><strong>CosWeekly</strong></td>
<td> <em>Cos (day of the week / 7) </em>.</td>
</tr>

<tr>
<td><strong>Case</strong></td>
<td>Target</td>
<td> COVID-19 infection at county level.</td>
<td><span><a href="https://usafacts.org/visualizations/coronavirus-covid-19-spread-map/" target="_blank">USA Facts</a></span></td>
</tr>
</tbody>
</table>

</div>

## Contribute

* Please do not add temporarily generated files in this repository.
* Make sure to clean your temp files before pushing any commits.
* In the .gitignore file you will find some paths in this directory are excluded from git tracking. So if you create anything in those folders, they won't be tracked by git.
  * To check which files git says untracked: `git status -u`. 
  * If you have folders you want to exclude, add the path in `.gitignore`, then `git add .gitignore`. Check again with `git status -u` if it is still being tracked.

## Citation
```
@INPROCEEDINGS{islam2023interpreting,
  author={Islam, Md Khairul and Liu, Yingzheng and Erkelens, Andrej and Daniello, Nick and Marathe, Aparna and Fox, Judy},
  booktitle={2023 IEEE International Conference on Digital Health (ICDH)}, 
  title={Interpreting County-Level COVID-19 Infections using Transformer and Deep Learning Time Series Models}, 
  year={2023},
  pages={266-277},
  doi={10.1109/ICDH60066.2023.00046}}
```