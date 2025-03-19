// Changing from normal dataset annotation to YOLO annotations

const fs = require("fs");
const path = require("path");

// Function to search files in a directory
function searchFilesInDirectory(directoryPath) {
  let results = [];

  // Read the directory
  const files = fs.readdirSync(directoryPath);
  files.forEach((file) => {
    const filePath = path.join(directoryPath, file);
    const stat = fs.statSync(filePath);

    // If it's a directory, recursively search it
    if (stat && stat.isDirectory()) {
      results = results.concat(searchFilesInDirectory(filePath));
    } else {
      results.push(filePath);
    }
  });

  return results;
}

const perFile = (ary) => {
  ary.forEach((element) => {
    const jsonData = JSON.parse(fs.readFileSync(element, "utf8"));
    const value = jsonData["objects"];
    const result = []; // result would be the coverted yolo annotation for each file and need to be saved as image name

    // const outputFilePath = path.join(path.dirname(element), 'output.txt');
    // const customString = "Your custom string goes here"; // Replace with your desired string
    // fs.writeFileSync(outputFilePath, customString, 'utf8');

    try {
      // console.log(value[0]["id"]);
      if (value.length < 1) {
        // If obejct is empty then it would be useless for us so we will skip that
        throw new TypeError("empty object");
      }
      // working on what is going to be saved
      const { height, width } = jsonData["size"];
      value.forEach((ele) => {
        const classId = ele["classId"];
      });

      // workig on how and where to be saved
      const name = element
        .split("\\")
        .pop()
        .replace(/\.json$/, "");
      console.log(name);
    } catch (e) {
      if (e instanceof TypeError) {
        console.log(e, value[0]);
      }
    }
  });
};

// Example usage
const directoryPath = "../../Datasets/pcbDataset/validation/ann/"; // Change this to the desired directory
const allFiles = searchFilesInDirectory(directoryPath);
console.log(allFiles.length);
perFile(allFiles);
