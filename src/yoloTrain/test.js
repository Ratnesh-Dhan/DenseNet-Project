const fs = require("fs");
const cliProgress = require("cli-progress");

const location = "../../Datasets/yoloPCB/labels/train/";
const searchFiles = fs.readdirSync(location);
// CLI PROGRESS
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
progressBar.start(searchFiles.length, 0);
let fileCounter = 0;
// CODE
searchFiles.forEach((e) => {
  const newFileName = e.replace(/\.jpg\.txt$/, ".txt");
  fs.renameSync(`${location}${e}`, `${location}${newFileName}`);
  // Updating file progressBar
  fileCounter++;
  progressBar.update(fileCounter);
});

progressBar.stop();
