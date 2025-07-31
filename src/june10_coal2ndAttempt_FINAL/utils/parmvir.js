const fs = require("fs");
const path = require("path");

const folder_path =
  "D:\\NML 2nd working directory\\DEEP SOUMYA 14-july-25\\final\\test";

const folders = fs.readdirSync(folder_path);

folders.forEach((element) => {
  let each_path = path.join(folder_path, element);
  let inside_folders = fs.readdirSync(each_path);
  let inside_path = path.join(each_path, inside_folders[0]);
  let inside_files = fs.readdirSync(inside_path);
  for (let i = 0; i < inside_files.length; i++) {
    fs.renameSync(
      path.join(inside_path, inside_files[i]),
      path.join(each_path, inside_files[i]),
      (error) => {
        if (error) {
          console.log(`❌ Error moving ${inside_files[i]}:`, error);
        } else {
          console.log(`✅ Moved: ${inside_files[i]} successfully`);
        }
      }
    );
  }
});
