class Test {
  constructor(name, age) {
    this.name = name;
    this.age = age;
  }

  show() {
    console.log(`from show() Name: ${this.name}, Age: ${this.age}`);
  }

  toString() {
    return `from toString() Name: ${this.name}, Age: ${this.age}`;
  }
}

const test = new Test("Happy", 18);
console.log(test);
console.log(test.toString());
