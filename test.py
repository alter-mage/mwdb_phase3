import pickle
with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
print(len(metadata.keys()))

f = 'image-smooth-14-7.png'
print(metadata[f].keys())