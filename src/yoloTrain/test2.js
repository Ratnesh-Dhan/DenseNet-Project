const fs = require("fs");
const path = require("path");

const directoryPath = "../../Datasets/pcbDataset/validation/ann/";
let file;
const files = fs.readdirSync(directoryPath);
files.forEach((element) => {
  console.log(element);
  file = path.join(directoryPath, element);
});

console.log(file);
const content = fs.readFileSync(file, "utf8");
console.log(content);
