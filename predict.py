import random

guess=random.randint(1, 100)

print("Welcome to the Number Guessing Game!")
print("I'm thinking of a number between 1 and 100.")
attempts=0
while True:
    attempts+=1
    choice=int(input("Enter number of choice: "))
    if choice>guess:
        print("The number is too high try again!")
    elif choice<guess:
        print("The number is too low try again!")
    else:
        print(f"Congratulations! You've guessed the number {guess} in {attempts} attempts.")
        break
    if attempts==10:
        print(f"Sorry, you've used all your attempts. The number was {guess}.")
        break
    