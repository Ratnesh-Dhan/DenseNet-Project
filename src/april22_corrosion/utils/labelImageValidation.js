const fs = require("fs");
const path = require("path");

const base_location =
  "C:\\Users\\NDT Lab\\Pictures\\dataset\\archive\\corrosion detect";

const image_files = fs.readdirSync(path.join(base_location, "/images"));

// Shuffle the array using Fisher-Yates algorithm
for (let i = image_files.length - 1; i > 0; i--) {
  const j = Math.floor(Math.random() * (i + 1));
  [image_files[i], image_files[j]] = [image_files[j], image_files[i]];
}

// Create validation directory if it doesn't exist
if (!fs.existsSync(path.join(base_location, "validation"))) {
  fs.mkdirSync(path.join(base_location, "validation"));
  fs.mkdirSync(path.join(base_location, "validation", "img"));
  fs.mkdirSync(path.join(base_location, "validation", "ann"));
}

for (let k = 0; k < 20; k++) {
  const sourceFile = path.join(base_location, "images", image_files[k]);
  const destinationFile = path.join(
    base_location,
    "validation",
    "img",
    image_files[k]
  );
  // Move the file
  fs.renameSync(sourceFile, destinationFile);

  const label_file = image_files[k].split(".")[0] + ".txt";
  const ann_source = path.join(base_location, "labels", label_file);
  const ann_destination = path.join(
    base_location,
    "validation",
    "ann",
    label_file
  );
  fs.renameSync(ann_source, ann_destination);
  console.log(k + 1);
}
