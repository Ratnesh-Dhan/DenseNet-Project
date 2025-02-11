from model import model
from data_loader import train_generator, val_generator, train_images, val_images

batch_size = 8
epochs = 10

history = model.fit(
    train_generator(),
    steps_per_epoch=len(train_images) // batch_size,
    epochs=epochs,
    validation_data=val_generator(),
    validation_steps=len(val_images) // batch_size
)

# Save the model
model.save('deeplabv3_plus_model.keras')