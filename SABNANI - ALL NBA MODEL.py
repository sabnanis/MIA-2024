#SHUBHAM SABNANI
#This model uses data from 2010 to 2018 with logistic regression
#in order to predict ALL-NBA selections for 2019
#The model correctly selected 13 of 15 team members, with the remaining two as runner ups

#IMPORTS--------------------------------------------------

#import os
#os.chdir('C:/Users/heliu/OneDrive/Documents/pythonbball')
#os.getcwd()
import pandas as pd
import numpy as np
from itertools import combinations
import statsmodels.api as sm
import matplotlib.pyplot as plt 
plt.rc("font", size=14)
import seaborn as sns
sns.set(style="white")
sns.set(style="whitegrid", color_codes=True)


#IMPORT PLAYER DATA-----------------------------------------

stats10 = pd.read_csv('playerstats10.csv', header=0)
stats11 = pd.read_csv('playerstats11.csv', header=0)
stats12 = pd.read_csv('playerstats12.csv', header=0)
stats13 = pd.read_csv('playerstats13.csv', header=0)
stats14 = pd.read_csv('playerstats14.csv', header=0)
stats15 = pd.read_csv('playerstats15.csv', header=0)
stats16 = pd.read_csv('playerstats16.csv', header=0)
stats17 = pd.read_csv('playerstats17.csv', header=0)
stats18 = pd.read_csv('playerstats18.csv', header=0)
stats = pd.concat([stats10,stats11,stats12,stats13,stats14,
                   stats15,stats16,stats17,stats18])
agestats = stats.copy()
stats = stats.loc[stats.MP >= 25] #Minimum threshold requirements for statistical significance
stats = stats.loc[stats.G >= 50]
stats = stats[['Player', 'Year', 'Pos', 'AllNBA', 'Age', 'MP', 'FG%', 'eFG%', 'FT%', 
         'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']] #Selecting test variables
agestats = agestats[['Player', 'Year', 'Pos', 'AllNBA', 'Age', 'MP', 'FG%', 'eFG%', 'FT%', 
         'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']]

stats['Pos'] = stats['Pos'].replace(['PG','SG'],'G')
stats['Pos'] = stats['Pos'].replace(['SF','PF'],'F')
agestats['Pos'] = agestats['Pos'].replace(['PG','SG'],'G')
agestats['Pos'] = agestats['Pos'].replace(['SF','PF'],'F')
#Limiting positions to be either 'G', 'F', or 'C'

#Test data -> 2018-2019 player averages
stats19 = pd.read_csv('playerstats19.csv', header=0)
stats19['Pos'] = stats19['Pos'].replace(['PG','SG'],'G')
stats19['Pos'] = stats19['Pos'].replace(['SF','PF'],'F')
forecast = stats19.copy() #For later use
stats19 = stats19.loc[stats19.MP >= 25]
stats19 = stats19.loc[stats19.G >= 50]
stats19 = stats19[['Player', 'Year', 'Pos', 'AllNBA', 'Age', 'MP', 'FG%', 'eFG%', 'FT%', 
         'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']]

stats = stats.fillna(0)
agestats = agestats.fillna(0)
stats19 = stats19.fillna(0)
forecast = forecast.fillna(0)


#ALL NBA PREDICTION MODEL (LOGISTIC REGRESSION)----------------

traincols = stats.columns[4:]
testcols = stats19.columns[4:]
logit = sm.Logit(stats['AllNBA'], stats[traincols])
result = logit.fit()
predictions = result.predict(stats19[testcols])
predictions
result.summary()

#Compiling the All-NBA team (predicted)
predictdf = pd.DataFrame({'AllNBA': predictions})
final = pd.concat([stats19[['Player','Pos']],predictdf],axis=1)
finalcopy = final.copy()
final = final.sort_values(by='AllNBA',ascending = False)
centers = final.loc[final.Pos == 'C'].head(3)
forwards = final.loc[final.Pos == 'F'].head(6)
guards = final.loc[final.Pos == 'G'].head(6)
team = pd.concat([guards,forwards,centers])
team = team.sort_values(by='AllNBA', ascending = False)


#FUTURE PLAYER STATISTICS PROJECTIONS---------------------------

dfs = list()
for i in range(19,36):
    for j in range(0,3):
        positions = ['G','F','C']
        temps = agestats.iloc[:,4:].loc[(agestats.Age == i)  & (agestats.Pos == positions[j])]
        temps.mean(axis=0)
        dfs.append(temps.mean(axis=0))

mean_matrix = pd.concat(dfs,axis=1).T
a = np.array(['G','F','C'])
mean_matrix = mean_matrix.assign(Pos=a[np.arange(len(mean_matrix)) % len(a)])
save = mean_matrix.iloc[3:,[0,11]]
mean_matrix = mean_matrix.set_index(['Age','Pos'])


dfs2 = list()
for i in range(3,len(mean_matrix)):
    current = mean_matrix.iloc[i,:]
    previous = mean_matrix.iloc[i-3,:]
    temp = (current - previous) / current #Average player percent-change for each statistic
    dfs2.append(temp)                       #Year-over-year

matrix2 = pd.concat(dfs2,axis=1).T
matrix2 = pd.concat([save.reset_index(),matrix2],axis=1)
matrix2 = matrix2.drop(['index'],axis=1)

length = len(forecast)
forecast = forecast[['Player', 'Year', 'Pos', 'Age', 'MP', 'FG%', 'eFG%', 'FT%', 
         'PTS', 'TRB', 'AST', 'STL', 'BLK', 'TOV']]
for i in range(0,length):
    for j in range(1, 36 - int(forecast.iloc[i,3])):
        forecast = forecast.append(pd.Series(),ignore_index=True)
        forecast.iloc[len(forecast)-1,[0,2]] = forecast.iloc[i,[0,2]]
        forecast.iloc[len(forecast)-1,[1,3]] = forecast.iloc[i,[1,3]]+j
        ageRef = int(forecast.iloc[i,3]+j)
        lastYear = forecast.loc[(forecast.Player == forecast.iloc[i,0]) & (forecast.Age == ageRef-1),:]
        agePos = matrix2.loc[(matrix2.Age == ageRef) & (matrix2.Pos == forecast.iloc[i,2])]
        forecast.iloc[len(forecast)-1,4:] = lastYear.iloc[0] * (1+ agePos.iloc[0,2:])
        
        
#ALL NBA PREDICTION MODEL APPLIED TO FUTURE STATISTICS---------------------
        
forecast = forecast.loc[(forecast.Year >=2020) & (forecast.MP >= 25)]
finalstats = pd.concat([stats,stats19])
traincols = finalstats.columns[4:]
testcols = forecast.columns[3:]
logit = sm.Logit(finalstats['AllNBA'], finalstats[traincols])
result = logit.fit()
predictions = result.predict(forecast[testcols])
predictions
result.summary()

#Compiling the All-NBA team (predicted)
predictdf2 = pd.DataFrame({'AllNBA': predictions})
final2 = pd.concat([forecast[['Player','Year','Pos']],predictdf2],axis=1)
final2 = final2.sort_values(by='AllNBA',ascending = False)
yearmax = int(final2.Year.max())
futureteams = pd.DataFrame()
for i in range(2020,yearmax+1):
    temp2 = final2.loc[final2.Year == i]
    centers = temp2.loc[temp2.Pos == 'C'].head(3)
    forwards = temp2.loc[temp2.Pos == 'F'].head(6)
    guards = temp2.loc[temp2.Pos == 'G'].head(6)
    futureteams = pd.concat([futureteams,guards,forwards,centers])
    futureteams = futureteams.sort_values(by=['Year','AllNBA'], ascending = True)

futureteams = futureteams.loc[futureteams.AllNBA >= 0.5]
futureteams['Player'].value_counts()


#FOUR PLAYERS PROBABILITY ANALYSIS-------------------------------------

interest = final2.loc[(final2.Player == 'Luka Doncic') | (final2.Player == 'Kyrie Irving') | (final2.Player == 'Stephen Curry') | (final2.Player == 'Karl-Anthony Towns')]
interest = interest.sort_values(by=['Player','Year'])
interest = interest.pivot(index='Player',columns='Year',values='AllNBA')
interest = interest.fillna(0)

#Finding the probability of EXACTLY N selections for each player
#Will take a couple minutes to compile
num_cols = len(interest.columns)
prob1 = list(range(0,num_cols))
exactprob = pd.DataFrame()
for i in range(0,len(interest)):
    list1 = []
    for j in range(0,num_cols+1):
        combos = list(combinations(prob1,j))
        count = 0
        for k in range(0,len(combos)):
            temp = 1
            for m in range(0,num_cols):
                if m in combos[k]:
                    temp = temp * interest.iloc[i,m]
                else:
                    temp = temp * (1 - interest.iloc[i,m])
            count = count + temp
        list1.append(count)
    exactprob = exactprob.append(pd.Series(list1), ignore_index = True)
exactprob.columns = list(range(0,num_cols+1))

#Finding the probability of AT MOST N selections for each player
atmostprob = exactprob.copy()
for i in range(0,len(atmostprob)):
    for j in range(1,len(atmostprob.columns)):
        atmostprob.iloc[i,j] += atmostprob.iloc[i,j-1]
        
#Calculating probability of having most selections out of the group
finalFrame = pd.DataFrame()
list2 = []
for i in range(0,len(exactprob)):
    count = 0
    for j in range(1,len(exactprob.columns)):
        temp = exactprob.iloc[i,j]
        for k in range(0,len(atmostprob)):
            if k != i:
                temp = temp * atmostprob.iloc[k,j-1]
        count += temp
    list2.append((count*100))
    

finalFrame = finalFrame.append(pd.Series(list2), ignore_index = True)
finalFrame = finalFrame.T
finalFrame.columns = list(['Percentage'])
finalFrame = finalFrame.set_index(interest.index)


#VISUALIZATION INSIGHTS----------------------------------------------------

mean_matrix['PTS'].unstack().plot.line()
mean_matrix['TRB'].unstack().plot.line()
mean_matrix['AST'].unstack().plot.line()

interest.T.plot.area(stacked=False, color = ["#EF3B24","#002D62","#007AC1","#FDBB30"],title = "All-NBA Odds by Year")

exactprob.T.plot.area()

