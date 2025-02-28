const fs = require("fs");

function readJsonFile(filePath) {
  try {
    const data = JSON.parse(fs.readFileSync(filePath, "utf8"));

    // Writing label_map.pbtxt file
    const labelMapContent = data.classes
      .map((element) => {
        return `item {\n id: ${element.id} \n name: '${element.title}' \n}`;
      })
      .join("\n");

    fs.writeFileSync("label_map.pbtxt", labelMapContent, "utf8");

    // return JSON.parse(data);
  } catch (error) {
    console.error("Error reading JSON file:", error);
    return null;
  }
}

// Example usage
readJsonFile("../../Datasets/pcbDataset/meta.json");
