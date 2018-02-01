import numpy as np
import pandas as pd

raw_data = np.genfromtxt("../emotion_seq.csv", delimiter=',')

reduced_data = [[]]
accumulation_data = [[]]
fps = 20
seconds = int(raw_data.shape[0] / fps)

for line in range(0, seconds):
    newline = np.mean(raw_data[fps * line + 1 : fps * (line + 1), :], axis=0).reshape(1, 7)
    if line == 0:
        reduced_data = newline
        accumulation_data = np.square(newline)
        oldline = newline
    else:
        reduced_data = np.append(reduced_data, newline, axis=0)
        newline = np.add(oldline, np.square(newline))
        accumulation_data = np.append(accumulation_data, newline, axis=0)
        oldline = newline

acc_percent_data = np.divide(accumulation_data, np.sum(accumulation_data, axis=1).reshape(seconds,1)) * 100


# temp1 = pd.Series(reduced_data.tolist())
# temp2 = temp1.to_json(orient='values')
# print(type(temp1))
# np.savetxt("eps.csv", reduced_data, delimiter=",", fmt="%1.2f")
np.savetxt("acc.csv", acc_percent_data, delimiter=",", fmt="%1.2f")