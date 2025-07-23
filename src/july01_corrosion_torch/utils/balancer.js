const fs = require("fs");
const path = require("path");

const base_path = "D:/NML 2nd working directory/dataset 2025-04-25 16-40-02";

const img_path = path.join(base_path, "img");
const maskHuman_path = path.join(base_path, "masks_human");
const image_path = path.join(base_path, "images");

if (!fs.existsSync(image_path)) {
  console.log("not exsits. then making the directory");
  fs.mkdirSync(image_path);
}

const maskHuman_files = fs.readdirSync(maskHuman_path);

for (let i = 0; i < maskHuman_files.length; i++) {
  try {
    fs.renameSync(
      path.join(img_path, maskHuman_files[i]),
      path.join(image_path, maskHuman_files[i])
    );
    console.log(`Done copying ${maskHuman_files[i]}`);
  } catch (error) {
    console.log(`There is an error: ${error}`);
  }
}
