# -*- coding: utf-8 -*-
"""
Created on Wed Sep 28 13:20:50 2022

@author: xgilbert
"""

##imports
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

random_state=42

#%%

year_n=2021
year_0=2015
window_n=3

# df_w=pd.read_csv('multi_test_full.csv')
df_all=pd.read_csv('data/eoy_wr_half.csv')
# adv_df=pd.read_csv('adv29test.csv')
# adv_df.loc[adv_df['% TM']=='%']='0'

# df_w=pd.read_csv('../data/wr_3_2015_2021_half.csv')
df_w=df_all.loc[(year_0 <= df_all['Year']) & (df_all['Year'] <= year_n) & (df_all['Week'] <= window_n)]
df_w['% TM']=[float(x.strip(' %')) for x in df_w['% TM']]


m1=(df_all['Year']>=2021) & (df_all['Week']==18)
m2=(df_all['Year']<2021) & (df_all['Week']==17)
df_eoy=df_all.loc[(m1) | (m2)]
#%%

##determine class status of every player/year combo

def eval_class(df_w, df_eoy, eoy_thresh, pos_thresh, g_thresh, d_thresh):
    df_compare=pd.DataFrame(columns=['Player', 'Pos Window', 'Pos EOY', 'Diff', 'Year', 'EOY_FPTS'])
    eligibles=df_eoy.loc[(df_eoy['G']>g_thresh) & (df_eoy[' FPTS']>eoy_thresh)]['Player']

    eoy_wk=df_eoy['Week'].max()
    w=df_w['Week'].max()

    target_class=pd.DataFrame(columns=['Player', 'Year'])

    for p in eligibles:
        for y in df_eoy['Year'].unique():
            location=df_w.loc[(df_w['Player']==p) & (df_w['Year']==y) & (df_w['Week']==w)]['Rank']
            if location.size==1:
                pos_w=location.item()
            eoy_location=df_eoy.loc[(df_eoy['Player']==p) & (df_eoy['Year']==y) & (df_eoy['Week']==eoy_wk)]['Rank']
            if eoy_location.size==1:
                pos_eoy=eoy_location.item()

            diff=pos_w-pos_eoy
            eoy_fpts=df_eoy.loc[(df_eoy['Player']==p) & (df_eoy['Year']==y)][' FPTS']

            if eoy_fpts.size==1:
                eoy_val=eoy_fpts.item()
            else:
                eoy_val=0
            df_compare=df_compare.append({'Player':p
                                          , 'Pos Window':pos_w
                                          , 'Pos EOY': pos_eoy
                                          , 'Diff':diff
                                          , 'Year':y
                                          # , 'Week':w
                                          , 'EOY_FPTS':eoy_val}
                                          , ignore_index=True
                                          )

    target_class=target_class.append(df_compare.loc[(df_compare['Diff']>=d_thresh)
                    & (df_compare['EOY_FPTS']>eoy_thresh)
                    & (df_compare['Pos Window']>pos_thresh)
                    #& (df_compare['Year']==y)
                    ][['Player', 'Year']])

    return target_class, df_compare
#%%

eoy_t=120
pos_t=15
g_t=8
diff_t=-3


target_class, df_compare =eval_class(df_w, df_eoy
                        , eoy_thresh=eoy_t
                        , pos_thresh=pos_t
                        , g_thresh=g_t
                        , d_thresh=diff_t
                        )
#%%

##only players above pos_t in week 1
def initiate_modeling_df(df_w, target_class):

    m1=(df_w['Week']==1) & (df_w['Rank']>=pos_t)
    df=df_w.loc[m1, ['Player', 'Year', 'Rank']].drop_duplicates(keep='first')
    for i in range(df.shape[0]):
        year=df.iloc[i]['Year']
        player=df.iloc[i]['Player']
        m2=(df_w['Week']==df_w['Week'].max()) & (df_w['Player']==player) & (df_w['Year']==year)
        m3=(df['Player']==player) & (df['Year']==year)
        if df_w.loc[m2, ' FPTS'].size==1:
            val=df_w.loc[m2, ' FPTS'].item()
        else:
            val=0
        df.loc[m3, 'W_FPTS']=val

    df['class']=0

    for i in range(target_class.shape[0]):
        year=target_class.iloc[i]['Year']
        player=target_class.iloc[i]['Player']
        #week=target_class.iloc[i]['Week']
        m=(df['Player']==player) & (df['Year']==year)
        df.loc[m, 'class']=1

    return df

df=initiate_modeling_df(df_w, target_class)
#%%
wow_feats=['REC', '% TM', 'YAC']
win_feats=['TGT']

# window=df_w['Week'].max()

def add_features(df, df_w, win_feats, wow_feats):

    ##only need window week data
    for f in win_feats:
        df[f]=0
        for y in df_w['Year'].unique():
            for p in df['Player'].unique():
                m_w=(df_w['Player']==p) & (df_w['Year']==y) & (df_w['Week']==df_w['Week'].max())
                val=df_w.loc[m_w, f]
                m=(df['Player']==p) & (df['Year']==y)
                if val.size==1:
                    df.loc[m, f]=val.item()
                else:
                    df.loc[m, f]=0

    ##week over window-week data for features based on window week value
    for f in wow_feats:
        for y in df_w['Year'].unique():
            for p in df['Player'].unique():
                m=(df['Player']==p) & (df['Year']==y)
                m_win=(df_w['Player']==p) & (df_w['Year']==y) & (df_w['Week']==df_w['Week'].max())
                win_val=df_w.loc[m_win, f]
                if win_val.size==1:
                    win_val=win_val.item()
                else:
                    win_val=0
                df.loc[m, str(f+'_w'+str(df_w['Week'].max()))]=win_val

                for_mean=0
                for_pc_mean=0
                for w in range(1, df_w['Week'].max()):
                    m_wow=(df_w['Player']==p) & (df_w['Year']==y) & (df_w['Week']==w)

                    ##percentage change for previous weeks, value for window week
                    temp_val=df_w.loc[m_wow, f]

                    if temp_val.size==1:
                        val=temp_val.item()
                        for_mean=for_mean+val
                        if val!=0:
                             val=(win_val-val)/val
                             for_pc_mean=for_pc_mean+val
                        else:
                            val=0
                    else:
                        val=0

                    col_name=f+'_w'+str(w)+'_pc'
                    df.loc[m, col_name]=val

                mean_col_name=f+'_mean'
                mean_pc_col_name=f+'_pc_mean'
                df.loc[m, mean_col_name]=for_mean/(df_w['Week'].max())
                df.loc[m, mean_pc_col_name]=for_pc_mean/(df_w['Week'].max()-1)

    return df

df=add_features(df, df_w, win_feats, wow_feats)

#%%
import seaborn as sns
sns.scatterplot(x='% TM_mean', y='W_FPTS', data=df, hue='class')

df_eda=pd.merge(df, df_eoy[['Player', 'Year', ' FPTS']], on=['Player', 'Year'])
#%%
sns.scatterplot(x='% TM_mean', y=' FPTS', data=df_eda, hue='class')
#%%
import matplotlib.pyplot as plt
df_corr=df_w.corr()
plt.figure(figsize=(12,8))
sns.heatmap(df_corr, cmap='icefire')
#%%
##set index
df=df.set_index(df.agg('{0[Player]}_{0[Year]}'.format, axis=1))

features=['W_FPTS'
          , 'TGT'
          , '% TM_pc_mean'
#          , 'YAC_mean'
          ]

X=df[features]
y=df['class']

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=random_state)

from sklearn.svm import SVC
svc=SVC(gamma='auto')

svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
y_test=pd.DataFrame(y_test)
y_test['pred']=y_pred

#%%

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=random_state)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=150, n_jobs=-1, verbose=1, random_state=random_state)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)

print(classification_report(y_test, y_pred))
y_test=pd.DataFrame(y_test)
y_test['pred']=y_pred
