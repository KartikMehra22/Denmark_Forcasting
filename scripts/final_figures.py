from pathlib import Path
import os,json
ROOT=Path(__file__).resolve().parents[1];OUT=ROOT/'results'/'controlled';FIG=OUT/'figures';FIG.mkdir(exist_ok=True)
os.environ['MPLCONFIGDIR']=str(ROOT/'.cache'/'matplotlib')
import numpy as np,pandas as pd,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.titleweight':'bold','figure.facecolor':'white','savefig.facecolor':'white'})
models=['previous_day','previous_week','ridge','lightgbm','patchtst_ensemble','chronos_bolt_tiny']
labels=['Previous day','Previous week','Ridge','LightGBM','PatchTST ensemble','Chronos Bolt tiny']
colors=['#9ba3ad','#c0c5cb','#45718e','#b17a39','#86729b','#18776e']
m=pd.read_csv(OUT/'metrics.csv');summary=[];regimes=[];boots=[]
for mode in ['hourly','quarter_hour']:
    fig,axs=plt.subplots(1,2,figsize=(10.8,3.25),layout='constrained')
    for ax,area in zip(axs,['DK1','DK2']):
        g=m[(m['mode']==mode)&(m.area==area)].set_index('model')
        vals=g.loc[models,'MAE'];yy=np.arange(len(models))
        ax.barh(yy,vals,color=colors,height=.63);ax.set_yticks(yy,labels);ax.invert_yaxis()
        for y,v in zip(yy,vals):ax.text(v+.5,y,f'{v:.1f}',va='center',fontsize=9)
        ax.set_xlim(0,max(vals)*1.17);ax.set(title=area,xlabel='MAE in EUR/MWh');ax.grid(axis='x',alpha=.15);ax.set_axisbelow(True)
    fig.savefig(FIG/f'{mode}_comparison.png',dpi=220);plt.close(fig)
    for area in ['DK1','DK2']:
        p=pd.read_csv(OUT/f'{mode}_{area}_predictions.csv',parse_dates=['time_utc'])
        cov=((p.actual>=p.chronos_q10)&(p.actual<=p.chronos_q90)).mean()
        pin=[]
        for q,col in [(.1,'chronos_q10'),(.5,'chronos_bolt_tiny'),(.9,'chronos_q90')]:
            e=p.actual-p[col];pin.append(np.maximum(q*e,(q-1)*e).mean())
        summary.append(dict(mode=mode,area=area,coverage80=cov,width80=(p.chronos_q90-p.chronos_q10).mean(),mean_pinball=np.mean(pin)))
        for regime,mask in [('All',np.ones(len(p),bool)),('Negative prices',p.actual<0),('Upper 5 percent',p.actual>=p.actual.quantile(.95))]:
            for mod in models:
                e=p.loc[mask,mod]-p.loc[mask,'actual'];regimes.append(dict(mode=mode,area=area,regime=regime,model=mod,n=len(e),MAE=e.abs().mean()))
        daily=p.groupby('date').apply(lambda z:pd.Series({m:(z[m]-z.actual).abs().mean() for m in models}),include_groups=False)
        rng=np.random.default_rng(42)
        for mod in models[2:]:
            a=(daily.previous_day-daily[mod]).values;n=len(a);est=[]
            for _ in range(2000):
                starts=rng.integers(n,size=int(np.ceil(n/7)));ix=np.concatenate([np.arange(k,k+7)%n for k in starts])[:n];est.append(a[ix].mean())
            lo,hi=np.quantile(est,[.025,.975]);boots.append(dict(mode=mode,area=area,model=mod,MAE_reduction=a.mean(),low95=lo,high95=hi))
# Fixed last seven days for examples, chosen by date rather than model performance.
fig,axs=plt.subplots(2,1,figsize=(10.8,4.4),layout='constrained',sharex=True)
for ax,area in zip(axs,['DK1','DK2']):
    p=pd.read_csv(OUT/f'quarter_hour_{area}_predictions.csv',parse_dates=['time_utc'])
    p=p[p.time_utc>=p.time_utc.max()-pd.Timedelta(days=7)]
    ax.fill_between(p.time_utc,p.chronos_q10,p.chronos_q90,color='#18776e',alpha=.16,label='Chronos 10–90% interval')
    ax.plot(p.time_utc,p.actual,color='#202d3c',lw=1,label='Actual')
    ax.plot(p.time_utc,p.chronos_bolt_tiny,color='#18776e',lw=.9,label='Chronos median')
    ax.set(title=area,ylabel='EUR/MWh');ax.grid(alpha=.12)
axs[0].legend(ncol=3,fontsize=8,loc='upper left');axs[-1].set_xlabel('Delivery time in UTC')
fig.savefig(FIG/'quarter_hour_example.png',dpi=220);plt.close(fig)
# Error by local delivery hour, no pooling across zones.
fig,axs=plt.subplots(1,2,figsize=(10.8,3),layout='constrained',sharey=True)
for ax,area in zip(axs,['DK1','DK2']):
    p=pd.read_csv(OUT/f'hourly_{area}_predictions.csv',parse_dates=['time_utc']);hour=p.time_utc.dt.tz_convert('Europe/Copenhagen').dt.hour
    for mod,lab,col in [('previous_day','Previous day','#9ba3ad'),('ridge','Ridge','#45718e'),('chronos_bolt_tiny','Chronos','#18776e')]:
        vals=(p[mod]-p.actual).abs().groupby(hour).mean();ax.plot(vals.index,vals,label=lab,color=col,lw=1.6)
    ax.set(title=area,xlabel='Local delivery hour',xticks=[0,6,12,18,23]);ax.grid(alpha=.15)
axs[0].set_ylabel('MAE in EUR/MWh');axs[1].legend(fontsize=8)
fig.savefig(FIG/'hourly_error_profile.png',dpi=220);plt.close(fig)
pd.DataFrame(summary).to_csv(OUT/'uncertainty.csv',index=False)
pd.DataFrame(regimes).to_csv(OUT/'regime_metrics.csv',index=False)
pd.DataFrame(boots).to_csv(OUT/'bootstrap.csv',index=False)
print(pd.DataFrame(summary).to_string(index=False))
