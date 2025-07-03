const fs = require("fs");
const path = require("path");

const basePath =
  "D:/NML 2nd working directory/corrosion sample piece/dataset/train";

const base = "D:/NML 2nd working directory/corrosion sample piece/dataset";

const image_path = path.join(basePath, "images");
const annotation_path = path.join(basePath, "annotations");

const shuffleArray = (array) => {
  for (let i = array.length - 1; i > 0; i--) {
    const j = Math.floor(Math.random() * (i + 1));
    [array[i], array[j]] = [array[j], array[i]];
  }
  return array;
};

const image_files = fs.readdirSync(image_path);

const shuffled_image_files = shuffleArray(image_files);

const new_train_path = path.join(base, "train_reduced");
if (!fs.existsSync(new_train_path)) {
  fs.mkdirSync(new_train_path);
}

const new_train_image_path = path.join(new_train_path, "images");
if (!fs.existsSync(new_train_image_path)) {
  fs.mkdirSync(new_train_image_path);
}

const new_train_annotation_path = path.join(new_train_path, "annotations");
if (!fs.existsSync(new_train_annotation_path)) {
  fs.mkdirSync(new_train_annotation_path);
}

for (let i = 0; i < shuffled_image_files.length / 3; i++) {
  const ann_name = shuffled_image_files[i].replace(".jpg", "");

  fs.copyFileSync(
    path.join(image_path, shuffled_image_files[i]),
    path.join(new_train_image_path, shuffled_image_files[i])
  );
  fs.cpSync(
    path.join(annotation_path, ann_name),
    path.join(new_train_annotation_path, ann_name),
    { recursive: true }
  );
}
