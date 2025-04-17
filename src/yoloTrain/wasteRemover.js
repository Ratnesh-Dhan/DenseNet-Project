const fs = require("fs");
const path = require("path");

const purger = (directorylabel, directoryImage) => {
  const files = fs.readdirSync(directorylabel);
  const ary = [];
  files.forEach((file) => {
    const mod_file_name = file.replace(/\.txt$/, ".jpg"); // Trim the trailing '.txt' substring
    ary.push(mod_file_name);
  });
  //   console.log(ary);

  const imageFiles = fs.readdirSync(directoryImage);

  const result = imageFiles.filter((e) => !ary.includes(e));
  console.log(result);

  result.forEach((file) => {
    const filePath = path.join(directoryImage, file);
    fs.unlinkSync(filePath); // Delete the file
    console.log(`Deleted file: ${filePath}`);
  });
};

const directorylabel = "../../Datasets/yoloPCB/labels/train/";
const directoryImage = "../../Datasets/yoloPCB/images/train/";
purger(directorylabel, directoryImage);
