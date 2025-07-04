const fs = require("fs");
const path = require("path");

const basePath = "D:/NML 2nd working directory/corrosion sample piece/dataset";
const imageDir = path.join(basePath, "images");
const annotationDir = path.join(basePath, "annotations");

const trainImageDir = path.join(basePath, "train/images");
const trainAnnoDir = path.join(basePath, "train/annotations");
const valImageDir = path.join(basePath, "val/images");
const valAnnoDir = path.join(basePath, "val/annotations");

const fsExtra = require("fs-extra");

function ensureDirs(...dirs) {
  dirs.forEach((dir) => fsExtra.ensureDirSync(dir));
}

function getRandomSplit(items, valRatio = 0.01) {
  const shuffled = items.sort(() => 0.5 - Math.random());
  const valCount = Math.max(1, Math.floor(items.length * valRatio)); // at least 1
  return {
    val: shuffled.slice(0, valCount),
    train: shuffled.slice(valCount),
  };
}

function copyImageAndMask(name, destImageDir, destAnnoDir) {
  const imgSrc = path.join(imageDir, name + ".jpg");
  const annoSrc = path.join(annotationDir, name);

  const imgDst = path.join(destImageDir, name + ".jpg");
  const annoDst = path.join(destAnnoDir, name);

  if (fs.existsSync(imgSrc) && fs.existsSync(annoSrc)) {
    fsExtra.copySync(imgSrc, imgDst);
    fsExtra.copySync(annoSrc, annoDst);
  } else {
    console.warn(`⚠️ Skipping missing image or annotation: ${name}`);
  }
}

const main = () => {
  ensureDirs(trainImageDir, trainAnnoDir, valImageDir, valAnnoDir);

  const imageFiles = fs.readdirSync(imageDir).filter((f) => f.endsWith(".jpg"));
  const imageNames = imageFiles.map((f) => path.parse(f).name);

  const { train, val } = getRandomSplit(imageNames, 0.14); // 14%

  console.log(`🟢 Total images: ${imageNames.length}`);
  console.log(`📦 Training set: ${train.length}`);
  console.log(`🧪 Validation set: ${val.length}`);

  train.forEach((name) => copyImageAndMask(name, trainImageDir, trainAnnoDir));
  val.forEach((name) => copyImageAndMask(name, valImageDir, valAnnoDir));

  console.log("✅ Dataset split complete.");
};

main();
