const fs = require("fs");

const output_dir1 =
  "D:/NML ML Works/corrosion all masks/dataset 2025-04-25 16-40-02/merged_masks";
const output_dir2 =
  "D:/NML ML Works/corrosion all masks/dataset 2025-04-25 16-40-02/img";

const last_remover = (arrays) => {
  const return_array = [];
  arrays.forEach((element) => {
    return_array.push(element.split(".")[0]);
  });
  return return_array;
};

let array_0 = fs.readdirSync(output_dir1);
let array_1 = last_remover(array_0).sort();
let array_00 = fs.readdirSync(output_dir2);
let array_2 = last_remover(array_00).sort();
console.log(array_1.length, array_2.length);

for (let i = 0; i < 49; i++) {
  array_2.pop(array_1[i]);
}
console.log(array_2);
// const new_array = array_1.filter((item) => !array_2.includes(item));
