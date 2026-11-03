from pathlib import Path
import sys
import numpy as np,pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
root=Path(__file__).resolve().parent.parent; sys.path.insert(0,str(root))
from ab_calibrate_hbv import load_data,create_model,simulate
out=root/'presentation'/'figures'; out.mkdir(exist_ok=True)
d,area=load_data(); m=create_model(d,area); p=pd.read_csv(root/'calibration_outputs/best_parameters.csv'); names=p.parameter.tolist(); base=p.best_value.to_numpy()
cases={'Baseline':[],'Snowmelt off':['snw_pmf','snw_amf'],'Channel outlets off':['urr_ndr','lrr_dre'],'Subsurface exports off':['urr_uct','lrr_lct']}; qs={}; st={}
for name,pars in cases.items():
 v=base.copy()
 for x in pars: v[names.index(x)]=0
 qs[name]=simulate(m,v,full_outputs=True); arr=m.get_outputs().copy(); labels=m.get_output_labels(); st[name]=pd.DataFrame({k:arr[:,i] for k,i in labels.items()},index=d.index)
obs=d.diso__ref.to_numpy(); colors=['#1c2833','#1874a5','#dd7136']; plt.rcParams.update({'font.size':11,'axes.spines.top':False,'axes.spines.right':False,'legend.fontsize':9})
def save(fig,name):
 for ax in fig.axes:
  if name != 'convergence':
   locator=mdates.AutoDateLocator(minticks=3,maxticks=6); ax.xaxis.set_major_locator(locator); ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))
 fig.savefig(out/(name+'.pdf'),bbox_inches='tight'); plt.close(fig)
def lines(name,case=None,zoom=False):
 fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained'); ax.plot(d.index,obs,color=colors[0],lw=1.1,label='Observed'); ax.plot(d.index,qs['Baseline'],color=colors[1],lw=1.3,label='Baseline')
 if case: ax.plot(d.index,qs[case],color=colors[2],lw=1.3,label=case)
 if zoom: ax.set_xlim(pd.Timestamp('2011-01-10'),pd.Timestamp('2011-01-20'))
 ax.set_ylabel('Discharge (m³/s)'); ax.grid(alpha=.18); ax.legend(frameon=False,ncol=3,loc='upper right'); save(fig,name)
lines('baseline'); lines('peak_zoom',zoom=True)
for n,c in [('snow','Snowmelt off'),('outlets','Channel outlets off'),('subsurface','Subsurface exports off')]: lines(n,c)
lines('snow_zoom','Snowmelt off',True)
for name,col,case,ylabel in [('snow_storage','snw_dth','Snowmelt off','Snow storage (mm)'),('lower_storage','lrr_dth','Channel outlets off','Lower storage (mm)')]:
 fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained'); ax.plot(d.index,st['Baseline'][col],color=colors[1],label='Baseline'); ax.plot(d.index,st[case][col],color=colors[2],label=case); ax.set_ylabel(ylabel); ax.legend(frameon=False); ax.grid(alpha=.18); save(fig,name)
fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained'); ax.plot(d.index,qs['Subsurface exports off']-qs['Baseline'],color=colors[2]); ax.axhline(0,color='black',lw=.7); ax.set_ylabel('Change in discharge (m³/s)'); ax.grid(alpha=.18); save(fig,'subsurface_difference')
h=pd.read_csv(root/'calibration_outputs/evaluation_history.csv'); fig,ax=plt.subplots(figsize=(10,3.8),layout='constrained'); ax.plot(h.evaluation,h.best_so_far,color=colors[1]); ax.set_yscale('log'); ax.set_xlabel('Objective evaluations'); ax.set_ylabel('Best so far: 1 − NSE'); ax.grid(alpha=.2); save(fig,'convergence')
print('maximum absolute subsurface difference',np.max(np.abs(qs['Subsurface exports off']-qs['Baseline'])))
