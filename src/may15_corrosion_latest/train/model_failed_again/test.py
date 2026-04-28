with open('./best_model_transferLearning_test.h5', 'rb') as f:
    header = f.read(8)
    print(header)
