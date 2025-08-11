const fs = require("fs");
const path = require("path");

const base_path = "../../../../Datasets/Asirra_cat_vs_dogs";

const files = fs.readdirSync(base_path);
const image_files = [];
const xml_files = [];

const trainDir = path.join(base_path, "train");
if (!fs.existsSync(trainDir)) {
  fs.mkdirSync(trainDir);
}
const validDir = path.join(base_path, "validation");
if (!fs.existsSync(validDir)) fs.mkdirSync(validDir);

files.forEach((elements) => {
  if (elements.endsWith(".jpg")) {
    image_files.push(elements);
  } else if (elements.endsWith(".xml")) {
    xml_files.push(elements);
  }
});

const dog = [];
const cat = [];
image_files.forEach((element) => {
  if (element.startsWith("dog")) dog.push(element);
  else cat.push(element);
});

const mover = (file) => {
  const old_path = path.join(base_path, file);
  const new_path = path.join(trainDir, file);
  fs.renameSync(old_path, new_path);

  // for xml file
  const name_array = file.split(".");
  const new_xml_name = `${name_array[0]}.${name_array[1]}.xml`;
  const xml_old_path = path.join(base_path, new_xml_name);
  const xml_new_path = path.join(validDir, new_xml_name);
  fs.renameSync(xml_old_path, xml_new_path);
};

const eighty = 550 * 0.8;
for (let i = 0; i <= eighty; i++) {
  mover(cat[i]);
  mover(dog[i]);
}

console.log("done");
