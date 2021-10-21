import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn
from scipy.stats import poisson,skellam
import statsmodels.formula.api as smf
import datetime
import statsmodels.api as sm

#mmmhh need a bunch of data for the algorithm
#don't know what variable names  to use but  imma go with alphabets
#fml
a = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/E0.csv")
b = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/E1.csv")
c = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/SC0.csv")
d = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/D1.csv")
e = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/D2.csv")
f  = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/I1.csv")
g = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/I2.csv")
h = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/SP1.csv")
i = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/SP2.csv")
j = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/F1.csv")
k = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/F2.csv")
l = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/N1.csv")
m = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/B1.csv")
n = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/P1.csv")
o = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/T1.csv")
p = pd.read_csv("https://www.football-data.co.uk/mmz4281/2021/G1.csv")
Data = pd.concat([a,b,c,d,e,f,g,h,i,j,k,l,m,n,o,p], axis=0,sort=True)
Data = Data[['HomeTeam', 'AwayTeam', 'FTHG', 'FTAG']]
Data = Data.rename(columns={'FTHG':'HomeGoals', 'FTAG':'AwayGoals'})
goal_model_data = pd.concat([Data[['HomeTeam', 'AwayTeam', 'HomeGoals']].assign(home=1).rename(
    columns={'HomeTeam':'team', 'AwayTeam':'opponent', 'HomeGoals':'goals'}),
                             Data[['AwayTeam', 'HomeTeam', 'AwayGoals']].assign(home=0).rename(
                                 columns={'AwayTeam':'team', 'HomeTeam':'opponent', 'AwayGoals':'goals'})])
poisson_model = smf.glm(formula="goals ~ home + team + opponent", data = goal_model_data,
                        family=sm.families.Poisson()).fit()
def simulate_match(foot_model, hometeam,awayteam, max_goals=10):
    home_goals_avg = foot_model.predict(pd.DataFrame(data={'team':hometeam,
                                                           'opponent': awayteam, 'home':1},
                                                     index=[1])).values[0]
    away_goals_avg = foot_model.predict(pd.DataFrame(data={'team':awayteam,
                                                           'opponent':hometeam, 'home':0},
                                                     index=[1])).values[0]
    team_pred = [[poisson.pmf(i, team_avg) for i in range(0, max_goals+1)] for team_avg in [home_goals_avg, away_goals_avg]]
    print(hometeam , "VS", awayteam)
    return(np.outer(np.array(team_pred[0]), np.array(team_pred[1])))
while True:
    try:
        
       Home = input("Enter home team: ")
       Away = input("enter away Team: ")
       predict = simulate_match(poisson_model,Home,Away,max_goals=10)
       print("HomeTeam :", np.sum(np.tril(predict, -1)) *100, "%")
       #probability of draw
       print("Draw: ", np.sum(np.diag(predict)) * 100, "%")
       #probality of awayTeam to win
       print("AwayTeam : ", np.sum(np.triu(predict, 1)) * 100, "%")
    except Exception as e:
        print("either the home team or the away team do not march our records please enter and try again")
        pass
   

