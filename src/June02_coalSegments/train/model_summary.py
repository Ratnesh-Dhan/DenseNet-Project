import tensorflow as tf

model = tf.keras.models.load_model("../models/EarlyStoppedBest09June.keras")
print(model.summary())

with open("./results/model_summary.txt", "w") as f:
    model.summary(print_fn=lambda x: f.write(x+"\n"))