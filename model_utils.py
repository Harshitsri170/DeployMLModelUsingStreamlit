import pickle

def load_model(name):
    with open(f"models/{name}.pkl", "rb") as f:
        model, acc = pickle.load(f)
    return model, acc
