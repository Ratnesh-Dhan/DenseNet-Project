// this is to copy selected image_annotations from the list of images which are present
// in the img/ folder

// const fs = require("fs");
import fs from "fs";

const base_dir = "../../Datasets/";
const directoryPath = base_dir + "testDataset/img/"; // Specify the folder pat

const filesReader = (path) => {
  return new Promise((resolve, reject) => {
    fs.readdir(path, (err, files) => {
      if (err) {
        return reject("Unable to scan directory: " + err);
      }
      resolve(files);
    });
  });
};

const img_array = await filesReader(directoryPath);
const source = base_dir + "PASCAL VOC 2012/train/ann/";
const dist = base_dir + "testDataset/ann/";
img_array.forEach((element) => {
  fs.copyFile(
    source + element + ".json",
    dist + element + ".json",
    fs.constants.COPYFILE_EXCL,
    (err) => {
      console.log("error on copy : " + err);
    }
  );
});
