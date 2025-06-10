const fs = require("fs");
const path = require("path");

const shuffle = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
};

const base_path =
  "D:\\NML ML Works\\TRAINING-20250602T050431Z-1-001\\working dataset";
const train_organic_path = path.join(base_path, "train", "organic");
const train_inorganic_path = path.join(base_path, "train", "inorganic");
const validation_organic_path = path.join(base_path, "validation", "organic");
const validation_inorganic_path = path.join(
  base_path,
  "validation",
  "inorganic"
);

const train_organic_files = shuffle(fs.readdirSync(train_organic_path));
const train_inorganic_files = shuffle(fs.readdirSync(train_inorganic_path));

const organic_transferable_size = train_organic_files.length * 0.1;
const inorganic_transferable_size = train_inorganic_files.length * 0.1;

for (let i = 0; i < organic_transferable_size; i++) {
  fs.renameSync(
    path.join(train_organic_path, train_organic_files[i]),
    path.join(validation_organic_path, train_organic_files[i])
  );
}

for (let i = 0; i < inorganic_transferable_size; i++) {
  fs.renameSync(
    path.join(train_inorganic_path, train_inorganic_files[i]),
    path.join(validation_inorganic_path, train_inorganic_files[i])
  );
}
