import pickle

def pickle_data(train_X,train_y):
    wizmann_data_augmented = open("data/wizmann_data_augmented","wb")
    pickle.dump(train_X,wizmann_data_augmented)
    wizmann_label_augmented = open("data/wizmann_label_augmented","wb")
    pickle.dump(train_y,wizmann_label_augmented)