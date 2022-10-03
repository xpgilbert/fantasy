# pos=input('Which position? (WR, RB, QB, TE) ').lower()
# year=int(input('Through what year? '))
# start_year=int(input('Which starting year? '))
# window=int(input('Through which week (1-18)? '))
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

random_state=42

### FUNCTIONS ###
def eval_class(df_w, df_eoy, eoy_thresh=False, pos_thresh=False, g_thresh=False, d_thresh=False):
    df_compare=pd.DataFrame(columns=['Player', 'Pos Window', 'Pos EOY', 'Diff', 'Year', 'EOY_FPTS'])
    
    if g_thresh and eoy_thresh:    
        eligibles=df_eoy.loc[(df_eoy['G']>g_thresh) & (df_eoy[' FPTS']>eoy_thresh)]['Player']

    elif g_thresh and eoy_thresh==False:
        eligibles=df_eoy.loc[(df_eoy['G']>g_thresh)]['Player']
    
    elif eoy_thresh and g_thresh==False:
        eligibles=df_eoy.loc[(df_eoy[' FPTS']>eoy_thresh)]['Player']

    else:
        eligibles=df_eoy['Player']
   
    eoy_wk=df_eoy['Week'].max()
    w=df_w['Week'].max()

    target_class=pd.DataFrame(columns=['Player', 'Year'])

    for p in eligibles:
        for y in df_eoy['Year'].unique():
            location=df_w.loc[(df_w['Player']==p) & (df_w['Year']==y) & (df_w['Week']==w)]['Rank']
            if location.size==1:
                pos_w=location.item()
            else:
                continue
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
    # m=pd.Series()
    # if pos_thresh:
    #     m = m & (df_compare['Pos Window']>pos_thresh)
    
    # if d_thresh:
    #     m = m & (df_compare['Diff']>=d_thresh)
    
    # if eoy_thresh:
    #     m = m & (df_compare['EOY_FPTS']>eoy_thresh)
    
    for x in [pos_thresh, d_thresh, eoy_thresh]:
        if x == False:
            x=0
            
    m= (df_compare['Pos Window']>pos_thresh) & (df_compare['Diff']>=d_thresh) & (df_compare['EOY_FPTS']>eoy_thresh)
    target_class=target_class.append(df_compare.loc[m][['Player', 'Year']])

    return target_class, df_compare

def initiate_modeling_df(df_w, target_class, pos_t):

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
#%%
### START UP ###

print('''
      Program will first ask to set basic parameters with the
      option to override defaults.  Then you can play around
      with modeling parameters to arrive at suggested players.
      ''')
      
print('''
      Please now set the following: \n
      Player position to model.
      Week through which you are evaluating.
      ''')

while True:
    pos=input('Which position? (WR, RB, QB, TE) ').lower()
    if pos in set(['wr', 'rb', 'qb', 'te']):
        break
    else:
        print('---not a position go back to soccer---')
        
while True:
    window_n=int(input('Through which week (1-18)? '))
    if window_n <= 18:
        break
    else:
        print('---not a week go back to soccer---')


print('''
      Default timing parameters are:
          *Model uses data through the most recent year.
          *Earliest year used to train the model is 2015
          *Uses all available weeks (model starting week is 1) \n
      ''')
while True:
    o_time=input('Override default timing parameters (Yes or No)? ').lower()
    if o_time in set(['yes', 'no']):     
        if o_time=='yes':
            while True:
                year_n=int(input('Through what year? '))
                if year_n in set(range(2009, 2023)):
                    break
                else:
                    print('---not a real year go back to soccer---')  
            while True:
                year_0=int(input('Which starting year (earliest is 2009)? '))
                if year_0 in set(range(2009, 2023)):
                    break
                else:
                    print('---not a real year go back to soccer---')
            while True:
                window_0=int(input('Which starting week (less than the through week)? '))
                if window_0 <= window_n:
                    break
                else:
                    print('---that week wont work---')  
            break
        
        elif o_time=='no':
            year_n=2021
            year_0=2009
            window_0=1
            scoring='HALF'
            break
    else:
        print('---not an acceptable answer, please try Yes or No---')

        
# while True:
#     scoring=input('Scoring (HALF, PPR, Standard)? ' ).upper()
#     if window in set(['HALF', 'PPR', 'STANDARD']):
#         break
#     else:
#         print('not a scoring go back to soccer')
scoring='HALF'

file_name='eoy_{pos}_{scoring}.csv'.format(pos=pos, scoring=scoring)
file_path='data/{file_name}'.format(file_name=file_name)
df_all=pd.read_csv(file_path)

obj_cols=df_all.select_dtypes('object').columns
for col in obj_cols.drop('Player'):
    df_all[col]=df_all[col].apply(lambda x: x.replace(',','') if (type(x)==str) and (',' in x) else x)
    df_all[col]=df_all[col].apply(lambda x: float(x.strip(' %')) if type(x)==str else x)   
    
y1=(year_0 <= df_all['Year']) & (df_all['Year'] <= year_n)
w1=(df_all['Week'] >= window_0) & (df_all['Week'] <= window_n)
df_w=df_all.loc[y1 & w1]

m1=(df_all['Year']>=2021) & (df_all['Week']==18)
m2=(df_all['Year']<2021) & (df_all['Week']==17)
df_eoy=df_all.loc[(m1) | (m2)]
#%%
print('''
      Default target-class parameters for training the model are:
          *Player must have at least 100 fpoints by the end of the regular season.
          *Player could not have started in the top 20 ranks.
          *Max (or min) difference between starting rank and EoS rank is -3.
          *Had to have played at least 8 games during the regular season. \n
      ''')
      
while True:
    o_class=input('Override default target-class parameters (Yes or No)? ')
    if o_class.lower() in set(['yes', 'no']):
        if o_class=='yes':
            while True:
                scenario=input('Are you uploading a custom list of target players (Yes or No)? ')
                if scenario.lower() in set(['yes', 'no']):
                    if scenario.lower()=='yes':
                        scen_path=input('Please enter the file path: ')
                        break
                    elif scenario.lower()=='no':
                        print('''
      If you don't want to use any of these parameters when
      training the model, type SKIP.
                              ''')
                        while True:
                            eoy_t=input('End of Season Fantasy Point Threshold? ')
                            if eoy_t.upper()=='SKIP':
                                eoy_t=False
                                break
                            elif int(eoy_t)>=0:
                                eoy_t=int(eoy_t)
                                break
                            else:
                                print('---not a real score---')  
                        while True:
                            pos_t=input('Which maximum rank? ')
                            if pos_t.upper()=='SKIP':
                                pos_t=False
                                break
                            elif int(pos_t)>0:
                                pos_t=int(pos_t)
                                break
                            else:
                                print('---not a rank---')
                        while True:
                            diff_t=input('Maximum difference in ranks (can be negative)? ')
                            if diff_t.upper()=='SKIP':
                                diff_t=False
                                break
                            elif int(diff_t)==int:
                                diff_t=int(diff_t)
                                break
                            else:
                                print('---not a real difference in ranks---')  
                        while True:
                            g_t=input('Minimum number of games played? ')
                            if g_t.upper()=='SKIP':
                                g_t=False
                                break
                            elif int(g_t)>=0:
                                g_t=int(g_t)
                                break
                            else:
                                print('---not a real number of games---')  
                        break
                    else:
                        print('---not an acceptable answer, please try Yes or No---')

        elif o_class=='no':
            eoy_t=100
            pos_t=15
            g_t=False
            diff_t=False
            break
    else:
        print('---not an acceptable answer, please try Yes or No---')

#%%

default_feats={
    'wr':['TGT', '% TM', 'W_FPTS']
    , 'rb':['Y/A', 'YDS']
    
    
    
    
    }


feat_list=default_feats[pos]
print('''
      Default features for this model are:
          {feat_list}
      '''.format(feat_list=feat_list))
      
      
      
win_feats=list(df_all.columns)
for x in ['Player', 'ROST', 'G', 'Week', 'Year', 'Rank']:
    win_feats.remove(x)
wow_feats=[]  ##must build in later
#%%
## execution

# df_w=pd.DataFrame()
# df_eoy=pd.DataFrame()
# if scen_path.exists():
#     target_class = pd.read_csv(scen_path)
# else:
target_class, x= eval_class(df_w, df_eoy
                        , eoy_thresh=eoy_t
                        , pos_thresh=pos_t
                        , g_thresh=g_t
                        , d_thresh=diff_t
                        )
#%%
df=initiate_modeling_df(df_w, target_class, pos_t)

print('Of {eligibles} eligible players, {class_n} are in the target class'.format(eligibles=df.shape[0], class_n=target_class.shape[0]))
#%%
df=add_features(df, df_w, win_feats, wow_feats)
#%%
##set index
df=df.set_index(df.agg('{0[Player]}_{0[Year]}'.format, axis=1))

features=['W_FPTS'
          , 'YDS'
          , 'Y/A'
          # , 'YAC_mean'
          ]

X=df[features]
y=df['class']
#%%
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=random_state)

from sklearn.svm import SVC
svc=SVC(gamma='auto')

svc.fit(X_train, y_train)
y_pred=svc.predict(X_test)

from sklearn.metrics import classification_report

print(classification_report(y_test, y_pred))
y_svc_test=pd.DataFrame(y_test)
y_svc_test['pred']=y_pred

#%%

X_train, X_test, y_train, y_test=train_test_split(X, y, test_size=0.2, random_state=random_state)

from sklearn.ensemble import RandomForestClassifier

rfc=RandomForestClassifier(n_estimators=150, n_jobs=-1, verbose=1, random_state=random_state)
rfc.fit(X_train, y_train)
y_pred=rfc.predict(X_test)

print(classification_report(y_test, y_pred))
y_rfc_test=pd.DataFrame(y_test)
y_rfc_test['pred']=y_pred
#%%