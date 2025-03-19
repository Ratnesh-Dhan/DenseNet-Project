const fs = require("fs");

const file = "../../Datasets/pcbDataset/meta.json";
const jsonData = JSON.parse(fs.readFileSync(file, "utf8"));
const classes = jsonData.classes;
const classMap = {};
let count = 0;
classes.forEach((element) => {
  classMap[element.id] = { title: element.title, num: count };
  count++;
});
fs.writeFileSync(
  "../../Datasets/pcbDataset/classMap.json",
  JSON.stringify(classMap, null, 2),
  "utf8"
);
