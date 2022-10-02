'''
Scraper for all positions

https://www.fantasypros.com/ for NFL player data.

WR - Week 1 to Week X
'''

import requests as r
from bs4 import BeautifulSoup as soup
import pandas as pd

pos=input('Which position? (WR, RB, QB, TE) ').lower()
year=int(input('Through what year? '))
start_year=int(input('Which starting year? '))
window=int(input('Through which week (1-18)? '))
#advanced=input('Advanced (Yes or No)? ').lower()

#if advanced=='no':
scoring=input('Scoring (HALF, PPR, Standard)? ' ).upper()

df_fs=pd.DataFrame() #normal
df_fa=pd.DataFrame() #advanced
yr_range=list(range(start_year, year+1))
wk_range=list(range(1, window+1))

##normal
for year in yr_range:
    for window in wk_range:
        
        ##account for additional week after 2021
        if (year < 2021) and (window > 17):
            continue
        
        url='https://www.fantasypros.com/nfl/stats/{pos}.php?year={year}&scoring={scoring}&range=custom&start_week=1&end_week={window}'.format(year=year, window=window, scoring=scoring, pos=pos)

        data=r.get(url)

        page_data=soup(data.text, 'html.parser')
        table=page_data.find_all('tbody')[0]

        col_data=page_data.find('thead')
        col_data=col_data.find_all('tr')
        feat_labels=[x for x in col_data[1].find_all('small')]
        cols=[]
        for z in feat_labels:
            cols.append(z.contents[0])

        player_labels=table.find_all('td', {'class':'player-label'})
        players=[]
        teams=[]
        for i in player_labels:
            players.append(str(i.find('a', {'class':'player-name'}).contents[0]))
            teams.append(str(i.contents[1]))
        teams=[x.strip('() ') for x in teams]

        soup_stats=[]
        for x in table.find_all('td', {'class':'center'}):
            soup_stats.append(str(x.contents[0]))
        stats=[]
        for y in range(0, len(soup_stats), len(cols)):
            stats.append(soup_stats[y:y+len(cols)])

        df=pd.DataFrame(stats)
        c=dict(zip(list(range(len(cols))), cols))
        df=df.rename(columns=c)
        df.insert(0, 'Rank', range(1, len(players)+1))
        df.insert(1, 'Player', players)
    #    df.insert(2, 'Team', teams) ##team doesn't matter after current year, FA otherwise
        df.insert(3, 'Week', window)
        df.insert(4, 'Year', year)

        df_fs=pd.concat([df, df_fs])

##advanced
for year in yr_range:
    for window in wk_range:
        
        ##account for additional week after 2021
        if (year < 2021) and (window > 17):
            continue
        
        url='https://www.fantasypros.com/nfl/advanced-stats-{pos}.php?year={year}&range=custom&start_week=1&end_week={window}'.format(year=year, window=window, pos=pos)

        data=r.get(url)

        page_data=soup(data.text, 'html.parser')
        table=page_data.find_all('tbody')[0]

        col_data=page_data.find('thead')
        feat_labels=[x for x in col_data.find_all('span')]
        cols=[]
        for z in feat_labels:
            cols.append(z.contents[0])

        player_labels=table.find_all('td', {'class':'player-label'})
        players=[]
        teams=[]
        for i in player_labels:
            players.append(str(i.find('a', {'class':'player-name'}).contents[0]))
            teams.append(str(i.contents[1]))
        teams=[x.strip('() ') for x in teams]

        soup_stats=[]
        for x in table.find_all('td', {'class':'center'}):
            soup_stats.append(str(x.contents[0]))
        stats=[]
        for y in range(0, len(soup_stats), len(cols)):
            stats.append(soup_stats[y:y+len(cols)])

        df=pd.DataFrame(stats)
        c=dict(zip(list(range(len(cols))), cols))
        df=df.rename(columns=c)
        df.insert(0, 'Rank', range(1, len(players)+1))
        df.insert(1, 'Player', players)
    #    df.insert(2, 'Team', teams) ##team doesn't matter after current year, FA otherwise
        df.insert(3, 'Week', window)
        df.insert(4, 'Year', year)

        df_fa=pd.concat([df, df_fa])

##remove duplicate columns
adv_cols=[col for col in df_fa.columns if col not in df_fs.columns]
##add back merging columns
for x in ['Player', 'Week', 'Year']:
    adv_cols.append(x)

df_f=pd.merge(df_fs, df_fa[adv_cols], on=['Player', 'Year', 'Week'])
# else:
#     print('Invalid. Go to sleep')
#     quit()

file_name=input('Name your dataset: ')
file_name=file_name+'.csv'
file_path='../data/'+file_name
df_f.to_csv(file_path, index=False)
