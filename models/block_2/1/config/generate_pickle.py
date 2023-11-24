import pickle
from sklearn.preprocessing import MinMaxScaler

data = [[0, 1], [0, 0], [1, 3], [2, 0]]
scaler = MinMaxScaler()
scaler.fit(data)
with open("model.pickle", "wb") as f:
    pickle.dump(scaler, f)