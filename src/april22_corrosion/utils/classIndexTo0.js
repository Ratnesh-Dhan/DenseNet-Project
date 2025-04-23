const fs = require("fs");
const path = require("path");

const base_location =
  "C:\\Users\\NDT Lab\\Pictures\\dataset\\archive\\corrosion detect";
const labels = path.join(base_location, "labels");
const files = fs.readdirSync(labels);

let file_path = base_location;
files.forEach((element) => {
  file_path = path.join(base_location, "labels", element);
  const file = fs.readFileSync(file_path, "utf8");
  
const lines = file.split('\n');
const newLines = lines.map(line => {
    if (line.trim() === '') return line;
    const parts = line.split(' ');
    parts[0] = '0'; // Change class index to 0
    return parts.join(' ');
});
fs.writeFileSync(file_path, newLines.join('\n'));
});
