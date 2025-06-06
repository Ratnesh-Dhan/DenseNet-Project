const fs = require("fs");
const path = require("path");

const shuffle = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
  j;
};

const base_path =
  "D:\\NML ML Works\\TRAINING-20250602T050431Z-1-001\\working dataset";
const train_cavity_path = path.join(base_path, "train", "cavity");
const validation_cavity_path = path.join(base_path, "validation", "cavity");

const train_cavity_files = shuffle(fs.readdirSync(train_cavity_path));

const cavity_transferable_size = train_cavity_files.length * 0.1;

for (let i = 0; i < cavity_transferable_size; i++) {
  fs.renameSync(
    path.join(train_cavity_path, train_cavity_files[i]),
    path.join(validation_cavity_path, train_cavity_files[i])
  );
}
