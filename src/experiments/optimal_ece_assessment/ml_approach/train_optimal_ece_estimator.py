import pickle

with (open(f'./data/20250423_024810.pkl', 'rb') as file):
    data = pickle.load(file)
    print(data)