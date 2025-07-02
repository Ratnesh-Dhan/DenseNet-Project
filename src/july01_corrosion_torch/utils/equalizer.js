const fs = require("fs");
const path = require("path");

const show_stats = () => {
  console.log(images.length);
  console.log(annotations.length);
};
const basePath =
  "D:/NML 2nd working directory/corrosion sample piece/augmented";
const annotationsDir = path.join(basePath, "annotations");
const annotations = fs.readdirSync(annotationsDir);
const imageDir = path.join(basePath, "images");
const images = fs.readdirSync(imageDir);

console.log(images);
const delete_able = images.filter((image) => image.endsWith(".png"));
console.log(delete_able);
show_stats();

// const new_images = [];
// for (let i = 0; i < annotations.length; i++) {
//   if (
//     images.includes(`${annotations[i]}.jpg`) ||
//     images.includes(`${annotations[i]}.png`)
//   ) {
//     new_images.push(images[i]);
//   }
// }

// new_images.forEach((image) => {
//   images.pop(`${image}.png`);
//   images.pop(`${image}.jpg`);
// });

// console.log(images);
// console.log(images.length);
