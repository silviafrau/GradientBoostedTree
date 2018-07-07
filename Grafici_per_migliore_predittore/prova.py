import pandas as pd
from sklearn.preprocessing import normalize
from sklearn.metrics import accuracy_score
import numpy as np
from sklearn.ensemble import GradientBoostingClassifier
import matplotlib.pyplot as plt
import csv


def plotCovariance(nuovo_acc, name_fig, dataset_name, accuratezza_media,coverage_percentage, n_under_thre, number_elements,final_accuracy):
    mdd = maxDrawdown(nuovo_acc)
    end = nuovo_acc[(len(nuovo_acc)-1)] - 100000
    plt.figure(figsize = (10,7))
    plt.grid(True)
    plt.title('Market = '+ dataset_name +' mdd = '+ str(round(mdd,4))+' end = '
              +str(round(end,4))+ ' end/mdd = ' + str(round(end/mdd,4)) + ' final_accuracy ' + str(final_accuracy) + '\n' + ' avg_accuracies = '
              + str(round(accuratezza_media,2)) + ' coverage = ' + str(round(coverage_percentage,3)) + '% '
              + ' accuracies under 50% = ' + str(n_under_thre) + ' n_predictors = ' + str(number_elements))
    plt.ylabel('Amount(USD)')
    plt.xlabel('Days')
    plt.gca().get_yaxis().get_major_formatter().set_scientific(False)
    plt.draw()
    plt.plot(nuovo_acc)
    plt.savefig(name_fig)
#---------------------------------------------------------------------------
df = pd.read_csv('values.csv')

decision = df['DECISIONE']
print(decision)
andamento = df['ANDAMENTO MERCATO']
diff = df['CLOSE-OPEN']

tmp = []
for element in decision:
    if element == 'BUY':
        tmp.append(1)
    elif element == 'SELL':
        tmp.append(-1)
    else:
        tmp.append(0)
print(tmp)
to_plot= 100000

tmp2 = []
for i in range(0,len(tmp)):
    if tmp[i] == andamento[i]:
        to_plot += 50*abs(diff[i])
    elif tmp[i] == 0:
        pass
    else:
        to_plot -= 50*abs(diff[i])
    tmp2.append(to_plot)
print(tmp2)

plt.figure()
plt.plot(tmp2)
plt.show()