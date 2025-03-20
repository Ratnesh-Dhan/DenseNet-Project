// Changing from normal dataset annotation to YOLO annotations
const cliProgress = require("cli-progress");
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

const perFile = (ary, save_dir) => {
  // Create new progress bar
  const progressBar = new cliProgress.SingleBar(
    {
      format:
        "Converting to YOLO format |{bar}| {percentage}% | {value}/{total} files",
      barCompleteChar: "\u2588",
      barIncompleteChar: "\u2591",
      hideCursor: true,
    },
    cliProgress.Presets.shades_classic
  );

  // Start progress bar
  progressBar.start(ary.length, 0);

  if (!fs.existsSync(save_dir)) {
    fs.mkdirSync(save_dir, { recursive: true });
  }
  // This map_data will help to get the integer for classId
  const map_data = JSON.parse(
    fs.readFileSync("../../Datasets/pcbDataset/classMap.json", "utf8")
  );

  ary.forEach((element) => {
    const jsonData = JSON.parse(fs.readFileSync(element, "utf8"));
    const value = jsonData["objects"];
    const result = []; // result would be the coverted yolo annotation for each file and need to be saved as image name

    // const outputFilePath = path.join(path.dirname(element), 'output.txt');
    // const customString = "Your custom string goes here"; // Replace with your desired string
    // fs.writeFileSync(outputFilePath, customString, 'utf8');

    try {
      // console.log(value[0]["id"]);
      if (Array.isArray(value) && value.length === 0) {
        // If the value is an empty array then it would be useless for us so we will skip that
        throw new TypeError("empty object");
      }
      // working on what is going to be saved
      const { height, width } = jsonData["size"];
      const single_yolo_file_array = [];
      value.forEach((ele) => {
        const classId = map_data[ele["classId"]].num;
        const xmin = ele["points"]["exterior"][0][0];
        const ymin = ele["points"]["exterior"][0][1];
        const xmax = ele["points"]["exterior"][1][0];
        const ymax = ele["points"]["exterior"][1][1];

        const x_center = (xmin + xmax) / 2.0 / width;
        const y_center = (ymin + ymax) / 2.0 / height;
        const yolo_width = Math.abs(xmax - xmin) / width;
        const yolo_height = Math.abs(ymax - ymin) / height;

        single_yolo_file_array.push(
          `${classId} ${x_center.toFixed(6)} ${y_center.toFixed(
            6
          )} ${yolo_width.toFixed(6)} ${yolo_height.toFixed(6)}`
        );
      });
      // console.log(single_yolo_file_array);
      // workig on how and where to be saved
      const name = element
        .split("\\")
        .pop()
        .replace(/\.json$/, "");
      // Writing file
      const outputPath = path.join(save_dir, `${name}.txt`);
      fs.writeFileSync(outputPath, single_yolo_file_array.join("\n"), "utf8");
    } catch (e) {
      if (e instanceof TypeError) {
        console.log(jsonData);
      } else {
        console.log("ERROR, : ", e);
      }
    }
  });
};

// Example usage
const variable_directory = "validation";
const directoryPath = `../../Datasets/pcbDataset/${variable_directory}/ann/`; // Change this to the desired directory
const save_directory = `../../Datasets/pcbDataset/YOLO/${variable_directory}/`;
const allFiles = searchFilesInDirectory(directoryPath);
console.log(allFiles.length);
perFile(allFiles, save_directory);
