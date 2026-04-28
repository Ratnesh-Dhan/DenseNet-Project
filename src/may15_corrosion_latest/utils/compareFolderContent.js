const fs = require("fs");

const path = require("path");

const original_dir =
  "D:/NML ML Works/corrosion all masks/dataset 2025-04-25 16-40-02/img_2nd_png_version";
const png_dir =
  "D:/NML ML Works/corrosion all masks/dataset 2025-04-25 16-40-02/filtered_corrosion_2nd_png_version";

let original_content = fs.readdirSync(original_dir);
let png_content = fs.readdirSync(png_dir);

const ext_remover_for_text_only = (arays) => {
  const new_aray = [];
  arays.forEach((element) => {
    new_aray.push(element.split(".")[0]);
  });
  return new_aray;
};

original_content = ext_remover_for_text_only(original_content);
png_content = ext_remover_for_text_only(png_content);

console.log(original_content.length);
console.log(png_content.length);
console.log(original_content.filter((item) => !png_content.includes(item)));
