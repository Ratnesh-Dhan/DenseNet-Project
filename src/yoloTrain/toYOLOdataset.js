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
    const name = element.split("\\").pop();
    console.log(name);
  });
};

// Example usage
const directoryPath = "../../Datasets/pcbDataset/validation/ann/"; // Change this to the desired directory
const allFiles = searchFilesInDirectory(directoryPath);
console.log(allFiles.length);
perFile(allFiles);
