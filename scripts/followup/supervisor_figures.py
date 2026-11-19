import sys as _sys
from pathlib import Path as _Path
_sys.path.insert(0, str(_Path(__file__).resolve().parents[1]))
from pathlib import Path
import os
R=Path(__file__).resolve().parents[2];O=R/'results/supervisor_response';F=O/'figures';F.mkdir(exist_ok=True);os.environ['MPLCONFIGDIR']=str(R/'.cache/matplotlib')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
rows=[]
for mode in ['hourly','quarter_hour']:
 for area in ['DK1','DK2']:
  p=pd.read_csv(R/f'results/completion/{mode}_{area}_predictions.csv');p['period']=p['split'].map({'historical':'earlier','later_locked':'later'})
  if mode=='hourly':p=p.merge(pd.read_csv(O/f'lear_weekly_{area}.csv')[['time_utc','lear_weekly']],on='time_utc',validate='one_to_one')
  w=pd.read_csv(O/f'winter_{mode}_{area}.csv');w['period']='winter'
  for frame in [p,w]:
   for per,g in frame.groupby('period'):
    models=['previous_day','ridge','lightgbm','patchtst','chronos']+(['lear_weekly'] if mode=='hourly' else [])
    for m in models:
     e=g[m]-g.actual;rows.append(dict(mode=mode,area=area,period=per,model=m,n=len(g),days=g.date.nunique(),MAE=e.abs().mean(),RMSE=np.sqrt((e**2).mean())))
M=pd.DataFrame(rows);M.to_csv(O/'summary_metrics.csv',index=False)
fig,axs=plt.subplots(2,2,figsize=(11,6),layout='constrained')
for ax,mode,area in zip(axs.flat,['hourly','hourly','quarter_hour','quarter_hour'],['DK1','DK2','DK1','DK2']):
 g=M[(M['mode']==mode)&(M.area==area)];models=['previous_day','ridge','lightgbm','patchtst','chronos']+(['lear_weekly'] if mode=='hourly' else [])
 labels=['Previous day','Ridge','LightGBM','PatchTST','Chronos']+(['LEAR weekly'] if mode=='hourly' else [])
 vals=np.array([[g[(g.model==m)&(g.period==p)].MAE.iloc[0] for p in ['winter','earlier','later']] for m in models]);im=ax.imshow(vals,aspect='auto',cmap='Blues',vmin=0,vmax=100)
 ax.set_xticks(range(3),['Winter 90d','Earlier '+('180d' if mode=='hourly' else '90d'),'Later 23d']);ax.set_yticks(range(len(models)),labels)
 for i in range(len(models)):
  for j in range(3):ax.text(j,i,f'{vals[i,j]:.1f}',ha='center',va='center',color='white' if vals[i,j]>58 else '#172b3a',fontsize=8)
 ax.set_title(('Hourly' if mode=='hourly' else '15 minute')+' '+area,weight='bold')
fig.savefig(F/'period_comparison.png',dpi=200);plt.close(fig)
h=pd.read_csv(O/'chronos_horizon.csv');fig,axs=plt.subplots(1,2,figsize=(11,3.2),layout='constrained',sharey=True)
for ax,area in zip(axs,['DK1','DK2']):
 for exp,color,label in [('original','#1d7874','July–November run'),('winter_independently_tuned','#b57931','Winter run')]:
  g=h[(h.area==area)&(h.experiment==exp)&(h.lead<=96)];ax.plot(g.lead,g.chronos_MAE,label=label,color=color,lw=1.7)
 ax.axvline(64,color='#777',ls='--',lw=1);ax.set_title(area,weight='bold');ax.set_xlabel('Quarter-hour forecast lead');ax.legend(fontsize=8);ax.grid(alpha=.15)
axs[0].set_ylabel('Chronos MAE (EUR/MWh)');fig.savefig(F/'chronos_lead.png',dpi=200);plt.close(fig)
print(M.to_string(index=False))
