const fs = require("fs");
const path = require("path");

const base_path = "../../../Datasets/NEU-DET/validation/images";

const all_folders = fs.readdirSync(base_path);
all_folders.forEach((element) => {
  let temp_image_path = path.join(base_path, element);
  let temp_files = fs.readdirSync(temp_image_path);
  temp_files.forEach((img) => {
    fs.renameSync(
      path.join(temp_image_path, img),
      path.join(base_path, img),
      (error = {
        if(error) {
          console.log("Error while moving : ", error);
        },
      })
    );
  });
});
