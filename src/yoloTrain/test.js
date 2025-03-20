const fs = require("fs");

const jsonData = JSON.parse(
  fs.readFileSync("../../Datasets/pcbDataset/classMap.json", "utf8")
);

const pint = jsonData["8337"].num;
console.log(typeof pint);
