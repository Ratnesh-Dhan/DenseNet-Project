const fs = require("fs");
const path = require("path");

const base_path = "../../../Datasets/NEU-DET";
const source_image_path = "../../../Datasets/NEU-DET/validation/images";
const source_annotaion_path =
  "../../../Datasets/NEU-DET/validation/annotations";
const total_split_images = 6;

const all_files = fs.readdirSync(source_image_path);

all_files.forEach((element) => {
  let temp_path = path.join(source_image_path, element);
  let temp_array = fs.readdirSync(temp_path);
  temp_array = shuffleArray(temp_array);
  let temp_test_path = path.join(base_path, "test", "images");
  let temp_annotation_path = path.join(base_path, "test", "annotations");

  fs.mkdirSync(temp_test_path, { recursive: true }, (error) => {
    if (error) {
      console.log("Error creating direcotry : ", error);
    } else {
      console.log("Created or already exsists.");
    }
  });
  fs.mkdirSync(temp_annotation_path, { recursive: true }, (error) => {
    if (error) {
      console.log("Error creating direcotry : ", error);
    } else {
      console.log("Created or already exsists.");
    }
  });
  temp_array.slice(0, total_split_images).forEach((img) => {
    fs.renameSync(
      path.join(temp_path, img),
      path.join(temp_test_path, img),
      (error) => {
        if (error) {
          console.log("Error while moving image : ", error);
        }
      }
    );
    let xml_filename = `${img.split(".")[0]}.xml`;
    fs.renameSync(
      path.join(source_annotaion_path, xml_filename),
      path.join(temp_annotation_path, xml_filename),
      (error) => {
        if (error) {
          console.log("Error while moving annotation : ", error);
        }
      }
    );
  });
});

function shuffleArray(array) {
  let currentIndex = array.length,
    randomIndex;

  // While there remain elements to shuffle.
  while (currentIndex !== 0) {
    // Pick a remaining element.
    randomIndex = Math.floor(Math.random() * currentIndex);
    currentIndex--;

    // And swap it with the current element.
    [array[currentIndex], array[randomIndex]] = [
      array[randomIndex],
      array[currentIndex],
    ];
  }

  return array;
}
