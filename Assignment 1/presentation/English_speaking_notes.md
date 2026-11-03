# Speaking notes: HBV calibration and process-switching experiments

These are suggested explanations, not a script that has to be read word for word. Allow roughly five minutes per speaker. If the 20-minute slot includes questions, shorten slides 15–17 and reserve about three minutes for discussion.

## Speaker 1: Question, calibration and baseline (slides 1–6; about 4–5 minutes)

**Slide 1 — Title**  
Today we will look at how the calibrated HBV model responds when we switch off selected hydrological processes. Our focus is on both the simulated discharge and where water is stored in the model.

**Slide 2 — Research question and roadmap**  
We start with a calibrated baseline. Then we disable three groups of pathways, one group at a time. We compare the resulting hydrographs and storage dynamics, and finish by discussing what the experiments can and cannot tell us.

**Slide 3 — Calibration setup**  
We use the supplied HBV001a model and hourly data from November 2010 to February 2011. The catchment area is 959 square kilometres. We calibrate 18 entries, including four initial storages, with Differential Evolution. Our objective is one minus NSE, so lower values indicate a better fit. Because the objective uses squared errors, larger discharge errors receive more weight.

**Slide 4 — Search history**  
Differential Evolution works with a population of parameter vectors. It generates trial vectors, runs HBV for each, and compares their objective values to decide which candidates to retain. Grey points show every evaluated candidate; the blue line shows the best value found so far. Individual trials may be worse, while the best-so-far line cannot increase. After the first candidate, the best value improved 61 times.

**Slide 5 — Stopping criterion**  
The plot divides the standard deviation of the population's objective values by the stopping threshold. The algorithm stops once this ratio reaches one or less. In generation 263 the population spread was approximately 0.00009095, below the threshold of 0.00009322. The final objective was 0.0930, equivalent to NSE 0.907, after 38,016 evaluations. This means the population's objective values became sufficiently close; it does not prove global optimality or uniqueness of the parameters.

**Slide 6 — Baseline performance**  
The calibrated baseline follows the observed hydrograph reasonably well overall. Its NSE is 0.907, RMSE is 7.86 cubic metres per second, and volume bias is minus 5.0 percent. There are still differences around some peaks. These statistics describe the calibration period: the entire record was used to fit the model, so they are not independent validation results.

## Speaker 2: Experiment design and snowmelt (slides 7–11; about 5 minutes)

**Slide 7 — Largest observed event**  
This zoom shows the largest observed peak. At the time of that observed peak, discharge is 153.5 cubic metres per second, while the baseline simulation is 105.1. We compare the two flows at the same time; the maximum simulated flow could occur at a different time.

**Slide 8 — Experiment design**  
For each experiment, we set a selected group of parameters to zero. The remaining parameters, initial storages and input data stay at their baseline values. We do not recalibrate. The three groups are snowmelt, selected reservoir outlets to the channel, and subsurface exports. This design isolates a change in pathways under the current calibrated parameter set.

**Slide 9 — Snowmelt off**  
We set both the rain-induced and air-induced snowmelt coefficients to zero. NSE falls from 0.907 to 0.414, and the volume bias becomes minus 43.0 percent. Several simulated events are smaller. This suggests that the baseline simulation depends substantially on those melt pathways under the current parameters.

**Slide 10 — Snowmelt and the January event**  
At the observed peak time, the model with snowmelt switched off simulates 68.1 cubic metres per second, compared with 105.1 for the baseline and 153.5 observed. The difference is the model's response to this intervention. We should not interpret it directly as a measurement of the real catchment's snowmelt contribution.

**Slide 11 — Snow storage**  
The model can still accumulate snowfall, but the two melt terms are zero. By the end of the record, snow storage reaches about 109.5 millimetres, versus zero in the baseline. We have therefore switched off snowmelt, not the entire snow module.

## Speaker 3: Reservoir outlets and subsurface exports (slides 12–15; about 5 minutes)

**Slide 12 — Selected channel outlets off**  
We set the upper reservoir's normal channel-outflow coefficient and the lower reservoir's channel-outflow coefficient to zero. The simulated channel discharge is zero throughout this experiment, and NSE is minus 0.754. A separate upper threshold outlet remains enabled, but it does not activate in this run. NSE can be negative when a simulation performs worse than the observed mean as a constant prediction, according to squared error.

**Slide 13 — Lower reservoir storage**  
The water does not simply disappear: much of it remains in the model. Final lower-reservoir storage reaches about 231.5 millimetres, compared with 2.4 millimetres in the baseline. This is an extreme pathway-switching experiment, so the storage plot helps explain why the hydrograph changed.

**Slide 14 — Subsurface exports off**  
Next we set the upper and lower subsurface export coefficients to zero. The discharge curves almost overlap, and NSE changes only from 0.906972 to 0.906997. Both baseline coefficients were already close to zero, so there was little export to remove under this parameter set.

**Slide 15 — A closer look at the difference**  
The difference plot makes the small change visible: the maximum absolute change in discharge is about 0.0233 cubic metres per second. Although subsurface exports are separate from the model's channel discharge, they affect stored water and can therefore influence later channel flow indirectly. A small effect here does not prove they are unimportant in all periods or parameter combinations.

## Speaker 4: Comparison, interpretation and discussion (slides 16–19; about 5 minutes)

**Slide 16 — Comparison**  
The first two interventions produce large changes in discharge and performance, whereas switching off subsurface exports has little effect in this run. NSE, RMSE and volume bias show different aspects of performance. NSE can be negative: zero corresponds to the squared error of a constant prediction equal to the observed mean.

**Slide 17 — Interpretation**  
The snowmelt experiment changes the release and storage of water in the snow component. The channel-outlet experiment disrupts the path to simulated discharge and increases storage. The subsurface experiment changes little because its baseline coefficients are near zero. These findings describe responses within this particular model and calibration; they are not direct measurements of process contributions in the real catchment.

**Slide 18 — Limitations and next steps**  
We have not recalibrated after disabling each pathway, so we do not know whether other parameters could compensate. We used the full record for calibration, without a warm-up or held-out validation period. NSE also emphasizes large squared errors, so peak timing, volume and low flows deserve separate attention. Finally, one converged DE run does not establish a unique or globally optimal parameter set.

**Slide 19 — Conclusions and questions**  
The baseline has a good overall fit, although it underestimates discharge at the largest observed peak time. Disabling snowmelt or selected channel outlets substantially changes discharge and storage; disabling subsurface exports barely changes this particular run. For discussion: should we recalibrate each simplified model? How should we select an independent validation period and additional performance metrics?
