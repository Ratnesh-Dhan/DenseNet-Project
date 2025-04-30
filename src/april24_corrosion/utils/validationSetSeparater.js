const fs = require("fs");
const path = require("path");

const base_location = "D:\\NML ML Works\\corrosion Final dataset";
const train_image_dir = path.join(base_location, "train/images");
const train_mask_dir = path.join(base_location, "train/corrosion_mask");
const train_piece_dir = path.join(base_location, "train/sample_piece_mask");
const files = fs.readdirSync(train_image_dir);

// radomizing the array
files.sort(() => Math.random() - 0.5);

const transfer_array = [];
for (i = 0; i < 60; i++) {
  transfer_array.push(files[i]);
}

const destinationFilePath1 = path.join(base_location, "validate/images");
const destinationFilePath2 = path.join(
  base_location,
  "validate/corrosion_mask"
);
const destinationFilePath3 = path.join(
  base_location,
  "validate/sample_piece_mask"
);

transfer_array.forEach((file) => {
  fs.rename(
    path.join(train_image_dir, file),
    path.join(destinationFilePath1, file),
    (err) => {
      if (err) {
        console.error("Error moving file:", err);
      } else {
        console.log("File moved successfully!");
      }
    }
  );
  fs.rename(
    path.join(train_mask_dir, file),
    path.join(destinationFilePath2, file),
    (err) => {
      if (err) {
        console.error("Error moving file:", err);
      } else {
        console.log("File moved successfully!");
      }
    }
  );
  fs.rename(
    path.join(train_piece_dir, file),
    path.join(destinationFilePath3, file),
    (err) => {
      if (err) {
        console.error("Error moving file:", err);
      } else {
        console.log("File moved successfully!");
      }
    }
  );
});
