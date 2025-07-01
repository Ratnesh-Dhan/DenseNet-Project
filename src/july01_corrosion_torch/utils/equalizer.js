const fs = require("fs");
const path = require("path");

const basePath =
  "D:/NML 2nd working directory/corrosion sample piece/augmented";
const annotationsDir = path.join(basePath, "annotations");
const annotations = fs.readdirSync(annotationsDir);
const imageDir = path.join(basePath, "images");
const images = fs.readdirSync(imageDir);
console.log(images.length);
console.log(annotations.length);
