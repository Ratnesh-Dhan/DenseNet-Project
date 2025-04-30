const fs = require("fs");
const path = require("path");

const base_location = "D:\\NML ML Works\\corrosion Final dataset\\train";
const corrosion_mask_dir = path.join(base_location, "corrosion_mask");
const piece_mask_dir = path.join(base_location, "sample_piece_mask");

const mask_files = fs.readdirSync(corrosion_mask_dir);
const piece_mask_files = fs.readdirSync(piece_mask_dir);

const mask_prefix = "corrosion_mask_";
const piece_prefix = "piece_mask_";

piece_mask_files.forEach((file) => {
  if (file.startsWith(piece_prefix)) {
    const newFileName = file.slice(piece_prefix.length);
    fs.rename(
      path.join(piece_mask_dir, file),
      path.join(piece_mask_dir, newFileName),
      (renameErr) => {
        if (renameErr) {
          console.error(`Error renaming file ${file}:`, renameErr);
        } else {
          console.log(`Renamed file: ${file} -> ${newFileName}`);
        }
      }
    );
  }
});

mask_files.forEach((file) => {
  if (file.startsWith(mask_prefix)) {
    const newFileName = file.slice(mask_prefix.length);
    fs.rename(
      path.join(corrosion_mask_dir, file),
      path.join(corrosion_mask_dir, newFileName),
      (renameErr) => {
        if (renameErr) {
          console.error(`Error renaming file ${file}:`, renameErr);
        } else {
          console.log(`Renamed file: ${file} -> ${newFileName}`);
        }
      }
    );
  }
});
