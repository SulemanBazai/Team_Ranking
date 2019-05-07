#This program uses the Page Rank Algorithm to determine the probability that a team in the NBA will win the championship based on data of the first few games. 
#To do this, an adjacency matrix A is constructed from the given data, and then converted into a normalzied transistion matrix M with column sum=1.
#Then, The power iteration algorithm is used to find the steady state vector x of the transition matrix M.
#This steady state vector is the predicted long-term probabilities of each team winning the Championship. The teams and their corresponding probabilities are graphed for visualization 
#Different data can be used to form a new prediction.
import numpy as np
import matplotlib.pyplot as plt
import copy
import scipy.linalg as la


#initial given data of teams and scores of games
home_team = np.array(['New York Knicks', 'San Antonio Spurs' ,'Utah Jazz', 'Brooklyn Nets'
 ,'Dallas Mavericks' ,'Houston Rockets', 'Minnesota Timberwolves',
 'Charlotte Hornets', 'Denver Nuggets' ,'Miami Heat' ,'Oklahoma City Thunder',
 'Sacramento Kings', 'Detroit Pistons', 'Washington Wizards',
 'Boston Celtics' ,'Los Angeles Clippers' ,'San Antonio Spurs',
 'Indiana Pacers', 'Houston Rockets' ,'Orlando Magic', 'Charlotte Hornets',
 'Golden State Warriors', 'Phoenix Suns' ,'Cleveland Cavaliers',
 'Los Angeles Lakers', 'Indiana Pacers' ,'Boston Celtics' ,'Orlando Magic',
 'Portland Trail Blazers' ,'Brooklyn Nets', 'Memphis Grizzlies',
 'Atlanta Hawks' ,'Minnesota Timberwolves' ,'New Orleans Pelicans',
 'Milwaukee Bucks', 'Dallas Mavericks', 'Utah Jazz' ,'Washington Wizards',
 'San Antonio Spurs' ,'Los Angeles Lakers', 'Golden State Warriors',
 'Sacramento Kings', 'Chicago Bulls' ,'Phoenix Suns' ,'Denver Nuggets',
 'Houston Rockets', 'New York Knicks' ,'Los Angeles Lakers',
 'Sacramento Kings', 'Memphis Grizzlies' ,'Milwaukee Bucks' ,'Orlando Magic',
 'Golden State Warriors', 'Utah Jazz', 'Los Angeles Lakers' ,'Chicago Bulls',
 'Detroit Pistons' ,'Philadelphia 76ers' ,'Oklahoma City Thunder',
 'New Orleans Pelicans', 'Houston Rockets', 'Portland Trail Blazers',
 'Dallas Mavericks' ,'Toronto Raptors', 'Boston Celtics',
 'Oklahoma City Thunder','Indiana Pacers' ,'Denver Nuggets',
 'Sacramento Kings', 'Charlotte Hornets' ,'New York Knicks',
 'Portland Trail Blazers' ,'Golden State Warriors' ,'Los Angeles Clippers',
 'Phoenix Suns', 'Miami Heat', 'San Antonio Spurs' ,'Atlanta Hawks',
 'Houston Rockets', 'Denver Nuggets', 'Chicago Bulls' ,'Sacramento Kings',
 'Minnesota Timberwolves' ,'Washington Wizards', 'Cleveland Cavaliers',
 'Los Angeles Clippers', 'Denver Nuggets' ,'Milwaukee Bucks' ,'Phoenix Suns',
 'Portland Trail Blazers', 'Utah Jazz' ,'Sacramento Kings' ,'Orlando Magic',
 'Indiana Pacers', 'New Orleans Pelicans' ,'Detroit Pistons' ,'Miami Heat',
 'Utah Jazz', 'Houston Rockets', 'Minnesota Timberwolves'])

away_team = np.array(['Cleveland Cavaliers', 'Golden State Warriors', 'Portland Trail Blazers'
 ,'Boston Celtics', 'Indiana Pacers', 'Los Angeles Lakers',
 'Memphis Grizzlies' ,'Milwaukee Bucks', 'New Orleans Pelicans',
 'Orlando Magic', 'Philadelphia 76ers', 'Phoenix Suns' ,'Toronto Raptors',
 'Atlanta Hawks','Chicago Bulls' ,'Portland Trail Blazers',
 'Sacramento Kings' ,'Brooklyn Nets' ,'Dallas Mavericks', 'Detroit Pistons',
 'Miami Heat', 'New Orleans Pelicans', 'Oklahoma City Thunder',
 'Toronto Raptors' ,'Utah Jazz' ,'Chicago Bulls' ,'Charlotte Hornets',
 'Cleveland Cavaliers', 'Denver Nuggets' ,'Milwaukee Bucks',
 'New York Knicks' ,'Philadelphia 76ers', 'Sacramento Kings',
 'San Antonio Spurs', 'Detroit Pistons', 'Houston Rockets',
 'Los Angeles Clippers', 'Memphis Grizzlies' ,'Miami Heat',
 'Oklahoma City Thunder' ,'Phoenix Suns', 'Atlanta Hawks' ,'Brooklyn Nets',
 'Los Angeles Clippers', 'Toronto Raptors' ,'Cleveland Cavaliers',
 'Detroit Pistons', 'Indiana Pacers', 'Miami Heat', 'Minnesota Timberwolves',
 'New Orleans Pelicans', 'Philadelphia 76ers', 'Portland Trail Blazers',
 'San Antonio Spurs' ,'Atlanta Hawks', 'Boston Celtics' ,'Brooklyn Nets',
 'Charlotte Hornets' ,'Los Angeles Clippers', 'Memphis Grizzlies',
 'New York Knicks', 'Phoenix Suns' ,'Utah Jazz' ,'Washington Wizards',
 'Cleveland Cavaliers' ,'Golden State Warriors', 'Milwaukee Bucks',
 'Minnesota Timberwolves' ,'Orlando Magic' ,'Brooklyn Nets' ,'Chicago Bulls',
 'Dallas Mavericks' ,'Los Angeles Lakers', 'Memphis Grizzlies',
 'New Orleans Pelicans', 'Toronto Raptors' ,'Utah Jazz', 'Washington Wizards',
 'Atlanta Hawks' ,'Detroit Pistons' ,'Indiana Pacers', 'Milwaukee Bucks',
 'Oklahoma City Thunder', 'Orlando Magic' ,'Philadelphia 76ers',
 'San Antonio Spurs','Boston Celtics', 'Dallas Mavericks',
 'Los Angeles Lakers' ,'Memphis Grizzlies', 'New York Knicks',
 'Toronto Raptors', 'Chicago Bulls', 'Charlotte Hornets',
 'Golden State Warriors', 'Los Angeles Clippers', 'Oklahoma City Thunder',
 'Philadelphia 76ers' ,'Washington Wizards' ,'Brooklyn Nets'])

home_score = np.array([88.,129.,104.,117.,121.,114.,98.,107.,107.,108.,103.,113.,91.,99.,99.,114.,102.,94.,106.,82.,97.,122.,110.,94.,89.,101.,104.,99.,115.,108.,104.,104.,103.,79.,83.,92.,75.,103.,106.,96.,106.,95.,118.,98.,102.,120.,89.,108.,96.,80.,117.,103.,127.,106.,123.,100.,101.,93.,85.,83.,118.,115.,81.,113.,122.,96.,107.,102.,94.,99.,117., 105. , 97. , 99., 112. , 87., 100. , 92., 97. , 86. , 94.,  91. , 92. , 86.,102. ,116., 123. , 75. ,108. , 100. ,114. , 96. , 80., 100., 106.,  82.,  85., 109.,114. ,110.])
away_score = np.array([117., 100., 113. ,122. ,130. ,120. ,102. , 96., 102.,  96.,  97. , 94., 109., 114.,105., 106. , 94. ,103. , 98. ,108. , 91., 114., 113. , 91. , 96. ,118.  ,98. ,105.,113., 110., 111. , 72., 106. , 98. , 98. , 93. , 88. ,112. , 99. ,113., 100. ,106.,88., 116., 105., 128. ,102., 115. ,108. ,116. ,113., 101. ,104. , 91. ,116., 107.,109. ,109. , 83. , 89. , 99., 118. , 97., 103., 128. ,122., 125. , 99., 102. , 95.,104. , 95. ,117. , 88., 111. , 96. , 86. , 95. ,112. ,103. ,111., 117., 112. , 88.,101. , 92. ,107. , 86., 119. , 94. ,109. , 91., 112., 122. ,116., 114. , 97. , 84.,106. ,119.])
#end of initial given data


#Actual program and algorithms start here

team_names = []    # list team names here

for i in range(len(away_team)):
    if home_team[i] in team_names:
        x = 0
    else:
        team_names.append(home_team[i])
#team names have been built into the list team_names

#initiate an adjacency matrix of proper shape
A = np.zeros(len(team_names)**2).reshape(len(team_names),len(team_names)) 
# the rows and columns should have the same order as your team_names

#build the adjacency matrix A with the representing scores data
#--an entry>0 means the team won by that many points, otherwise the entry is 0
for i in range(len(team_names)):
    row_team = team_names[i]
    for q in range(len(team_names)):
        column_team = team_names[q]
        for k in range(len(home_team)):
            if row_team==home_team[k] and column_team==away_team[k]:
                A[i,q] += max(0,home_score[k]-away_score[k])
            elif row_team==away_team[k] and column_team==home_team[k]:
                A[i,q] += max(0,away_score[k]-home_score[k])

#from adjacency matrix A, construct a transition matrix M that has each column sum=1
def transition(A):
    # function that constructs the matrix M
    M = A.copy()
    for i in range(len(A)):
        column_sum = np.sum(A[:,i])
        if column_sum != 0:
            M[:,i] = A[:,i]/column_sum
        else: 
            M[:,i] = 1/len(A)
    return M
    
M = transition(A)

#define a function that uses power iteration algorithm to find the steady state vector x for the given tolerance, or tol, and transition matrix M
def power_iteration(M,tol):
    x_0 = np.random.rand(len(M))
    xcurrent = x_0/np.linalg.norm(x_0,1)
    for i in range(1000000000000000000000):
        xtemp = M@xcurrent
        xtempnorm = np.linalg.norm(xtemp,1)
        xnext = xtemp/xtempnorm
        
        if np.linalg.norm(xnext-xcurrent,2)<tol:
            x = xnext
            break
        xcurrent = xnext
    return x
#define tol to be 1*10**(-8)
tol = 1*10**(-8)
x = power_iteration(M,tol)


#create a list called team_ranks that lists the order of most probable to win the championship to least probable to win using the data for the first given games
dict = {}
for i in range(len(x)):
    dict[team_names[i]] = x[i]

dict_sorted = sorted(dict.items(), key=lambda kv: kv[1])
dict_sorted.reverse()

team_ranks = []
for i in range(len(dict_sorted)):
    team_ranks.append(dict_sorted[i][0])
#print(team_ranks)

x.sort()
xsorted = x[::-1]


## Bar plot for the percentage of fan support for each team
## (following the order of team_ranks - "best" to "worst" team)
bar_height = xsorted
x_label = team_ranks


xaxis = range(len(team_ranks))
plt.bar(xaxis, bar_height)
plt.xticks(xaxis, x_label , rotation='vertical')
plt.subplots_adjust(bottom=0.4)
plt.title("Team vs Probability of Winning Championship")
plt.xlabel("Team Name")
plt.ylabel("Probability of Winning")
