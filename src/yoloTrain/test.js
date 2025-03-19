const fs = require("fs");

const jsonData = JSON.parse(
  fs.readFileSync("../../Datasets/pcbDataset/classMap.json", "utf8")
);

console.log(jsonData["8337"]);
