const fs = require("fs");
const path = require("path");

// Set the directory path and the prefix to remove
const directoryPath =
  "D:/NML ML Works/corrosion all masks/dataset 2025-04-25 16-40-02/filtered_corrosion_2nd_png_version"; // Replace with your folder path
const prefixToRemove = "corrosion_mask_"; // Replace with your actual prefix

// Read all files in the directory
fs.readdir(directoryPath, (err, files) => {
  if (err) {
    return console.error("Error reading directory:", err);
  }

  files.forEach((file) => {
    // Only rename if the file starts with the prefix
    if (file.startsWith(prefixToRemove)) {
      const oldPath = path.join(directoryPath, file);
      const newFileName = file.slice(prefixToRemove.length);
      const newPath = path.join(directoryPath, newFileName);

      // Rename the file
      fs.rename(oldPath, newPath, (err) => {
        if (err) {
          console.error(`Failed to rename ${file}:`, err);
        } else {
          console.log(`Renamed ${file} ➝ ${newFileName}`);
        }
      });
    }
  });
});
