# Q3 初步分析（K=3）
本报告基于 `outputs/q3_regression` 中已生成的 k=3（且评委侧采用 common-sample 对齐版）结果文件。
## 1) 量化影响（固定效应）
### 年龄（age_c）
- 评委：$\beta=-0.005$，95%HDI=[-0.006,-0.004]，每大 10 岁约乘以 0.951 (-4.9%)
- 粉丝：$\beta=-0.003$，95%HDI=[-0.005,-0.001]，每大 10 岁约乘以 0.970 (-3.0%)
- 差异（粉丝-评委）：$\Delta=+0.002$，95%HDI=[+0.000,+0.005]
解释：年龄系数为负表示“年龄越大，早期周内相对份额越低”。差异为正表示粉丝对年龄的惩罚更弱。
### 周次（log_week）
- 评委：$\beta=-0.011$，95%HDI=[-0.035,+0.014]
- 粉丝：$\beta=-0.010$，95%HDI=[-0.077,+0.057]
- 差异（粉丝-评委）：$\Delta=+0.000$，95%HDI=[-0.072,+0.070]
解释：当前 K=3 下，log_week 在两边都不算强信号（区间跨 0），说明早期周次趋势对“周内相对份额”解释有限或被随机效应吸收。
### 行业（industry）
行业系数是相对于一个基准行业（由建模时的编码自动确定，通常是频数最高的那类；常见情况是 Actor/Actress）。
- 评委（仅列出 95%HDI 不跨 0 的类别，按 |mean| 排序）：
  - industry[Model]: mean=-0.146, 95%HDI=[-0.206,-0.082], ratio≈0.864 (-13.6%)
  - industry[Other]: mean=-0.084, 95%HDI=[-0.145,-0.014], ratio≈0.919 (-8.1%)
  - industry[TV/Host]: mean=-0.065, 95%HDI=[-0.117,-0.020], ratio≈0.937 (-6.3%)
  - industry[Reality TV Star]: mean=-0.060, 95%HDI=[-0.091,-0.030], ratio≈0.942 (-5.8%)
  - industry[Athlete]: mean=-0.030, 95%HDI=[-0.058,-0.002], ratio≈0.970 (-3.0%)
- 粉丝（仅列出 95%HDI 不跨 0 的类别，按 |mean| 排序）：
  - industry[Model]: mean=-0.261, 95%HDI=[-0.421,-0.091], ratio≈0.770 (-23.0%)
  - industry[TV/Host]: mean=-0.168, 95%HDI=[-0.298,-0.040], ratio≈0.845 (-15.5%)

## 2) 差异性比较（固定效应差值 Δ=粉丝-评委_common）
下面按 |Δmean| 列出差异最大的若干项（K=3）。
- industry[Politician]: Δmean=-0.135（不显著）, 95%HDI=[-0.580,+0.296]
- industry[Model]: Δmean=-0.115（不显著）, 95%HDI=[-0.295,+0.056]
- industry[TV/Host]: Δmean=-0.103（不显著）, 95%HDI=[-0.237,+0.040]
- industry[Reality TV Star]: Δmean=+0.080（不显著）, 95%HDI=[-0.003,+0.169]
- industry[Musician]: Δmean=-0.078（不显著）, 95%HDI=[-0.167,+0.009]
- industry[Other]: Δmean=+0.044（不显著）, 95%HDI=[-0.148,+0.220]
- industry[Athlete]: Δmean=+0.029（不显著）, 95%HDI=[-0.046,+0.114]
- Intercept: Δmean=-0.003（不显著）, 95%HDI=[-0.082,+0.069]
- age_c: Δmean=+0.002（不显著）, 95%HDI=[+0.000,+0.005]
- log_week: Δmean=+0.000（不显著）, 95%HDI=[-0.072,+0.070]

## 3) 职业舞伴（Pro）随机效应：影响大小与一致性
- Pro 随机效应相关（common-sample）：json 给出 corr=0.27217480274515166, n_pairs=57；脚本按均值重算 corr=0.272（应接近）。
- 显著舞伴数量（95%HDI 不跨 0）：评委 2/57，粉丝 0/57（粉丝侧通常更不确定）。

### 评委侧：舞伴效应 Top/Bottom
- + Derek Hough: mean=+0.076, 95%≈[+0.028,+0.125], ratio≈1.079 (+7.9%)
- + Charlotte Jorgensen: mean=+0.056, 95%≈[-0.021,+0.157], ratio≈1.057 (+5.7%)
- + Artem Chigvintsev: mean=+0.047, 95%≈[-0.002,+0.100], ratio≈1.048 (+4.8%)
- + Valentin Chmerkovskiy: mean=+0.039, 95%≈[-0.005,+0.083], ratio≈1.040 (+4.0%)
- + Sasha Farber: mean=+0.026, 95%≈[-0.024,+0.077], ratio≈1.027 (+2.7%)
- + Kym Johnson: mean=+0.026, 95%≈[-0.023,+0.077], ratio≈1.026 (+2.6%)
- + Corky Ballas: mean=+0.024, 95%≈[-0.041,+0.097], ratio≈1.025 (+2.5%)
- + Tony Dovolani: mean=+0.024, 95%≈[-0.017,+0.066], ratio≈1.024 (+2.4%)

- - Ashly DelGrosso: mean=-0.065, 95%≈[-0.134,-0.002], ratio≈0.937 (-6.3%)
- - Elena Grinenko: mean=-0.063, 95%≈[-0.164,+0.014], ratio≈0.939 (-6.1%)
- - Koko Iwasaki: mean=-0.054, 95%≈[-0.138,+0.015], ratio≈0.947 (-5.3%)
- - Keo Motsepe: mean=-0.054, 95%≈[-0.117,+0.005], ratio≈0.947 (-5.3%)
- - Anna Demidova: mean=-0.041, 95%≈[-0.127,+0.033], ratio≈0.960 (-4.0%)
- - Peta Murgatroyd: mean=-0.038, 95%≈[-0.087,+0.010], ratio≈0.963 (-3.7%)
- - Edyta Sliwinska: mean=-0.036, 95%≈[-0.092,+0.015], ratio≈0.965 (-3.5%)
- - Chelsie Hightower: mean=-0.031, 95%≈[-0.090,+0.026], ratio≈0.970 (-3.0%)

### 粉丝侧：舞伴效应 Top/Bottom
- + Cheryl Burke: mean=+0.062, 95%≈[-0.021,+0.172], ratio≈1.064 (+6.4%)
- + Karina Smirnoff: mean=+0.043, 95%≈[-0.033,+0.145], ratio≈1.044 (+4.4%)
- + Sasha Farber: mean=+0.043, 95%≈[-0.044,+0.166], ratio≈1.044 (+4.4%)
- + Emma Slater/Kaitlyn Bristowe (week 9): mean=+0.037, 95%≈[-0.076,+0.206], ratio≈1.038 (+3.8%)
- + Maksim Chmerkoskiy: mean=+0.034, 95%≈[-0.046,+0.140], ratio≈1.035 (+3.5%)
- + Derek Hough: mean=+0.034, 95%≈[-0.046,+0.133], ratio≈1.035 (+3.5%)
- + Tristan MacManus: mean=+0.034, 95%≈[-0.073,+0.186], ratio≈1.035 (+3.5%)
- + Julianne Hough: mean=+0.033, 95%≈[-0.065,+0.171], ratio≈1.034 (+3.4%)

- - Keo Motsepe: mean=-0.064, 95%≈[-0.218,+0.033], ratio≈0.938 (-6.2%)
- - Jonathan Roberts: mean=-0.051, 95%≈[-0.218,+0.055], ratio≈0.950 (-5.0%)
- - Brian Fortuna: mean=-0.043, 95%≈[-0.229,+0.065], ratio≈0.957 (-4.3%)
- - Inna Brayer: mean=-0.042, 95%≈[-0.225,+0.071], ratio≈0.959 (-4.1%)
- - Henry Byalikov: mean=-0.041, 95%≈[-0.224,+0.071], ratio≈0.960 (-4.0%)
- - Pasha Pashkov: mean=-0.037, 95%≈[-0.169,+0.055], ratio≈0.964 (-3.6%)
- - Kym Johnson: mean=-0.035, 95%≈[-0.154,+0.053], ratio≈0.966 (-3.4%)
- - Britt Stewart: mean=-0.032, 95%≈[-0.157,+0.062], ratio≈0.968 (-3.2%)

### 舞伴‘粉丝 vs 评委’差异最大的个体（按 |mean_fan-mean_judge|）
- Ashly DelGrosso: fan=+0.013, judge=-0.065, |gap|=0.078
- Cheryl Burke: fan=+0.062, judge=-0.004, |gap|=0.066
- Kym Johnson: fan=-0.035, judge=+0.026, |gap|=0.060
- Karina Smirnoff: fan=+0.043, judge=-0.016, |gap|=0.059
- Anna Demidova: fan=+0.016, judge=-0.041, |gap|=0.057
- Pasha Pashkov: fan=-0.037, judge=+0.018, |gap|=0.055
- Jonathan Roberts: fan=-0.051, judge=-0.000, |gap|=0.051
- Charlotte Jorgensen: fan=+0.006, judge=+0.056, |gap|=0.050
- Brian Fortuna: fan=-0.043, judge=+0.005, |gap|=0.049
- Julianne Hough: fan=+0.033, judge=-0.015, |gap|=0.048

## 4) 一句话结论（供写进论文）
- 在 K=3 的早期周数据上，年龄对表现存在稳定负向影响，且评委侧惩罚略强于粉丝侧。
- 行业效应在评委侧更‘清晰’（更多类别显著偏离基准），粉丝侧行业差异不如评委侧稳定；但‘Model’在两边都显著偏低，且粉丝侧更低。
- Pro（职业舞伴）效应在两边仅弱相关（corr≈0.27），说明‘评委觉得某些舞伴能带来更高表现’与‘粉丝更愿意投某些舞伴搭档’并不完全一致。
