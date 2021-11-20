import pickle
with open('metadata.pickle', 'rb') as handle:
        metadata = pickle.load(handle)
print(metadata.keys())

# f = 'image-smooth-14-7.png'
# x_label, y_label, z_label = f.split('.')[0].split('-')[1:]
# print(x_label, y_label, z_label)