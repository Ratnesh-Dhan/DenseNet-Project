class Test:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"from __str__() Name: {self.name}, Age: {self.age}"

    def show(self):
        print(f"from show() Name: {self.name}, Age: {self.age}")

#object = instance of class
phani = Test('Phani', 21)
parmvir = Test('Parmvir', 20)

print(phani)
print(parmvir)








