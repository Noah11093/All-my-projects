print("Welcome to my computer quiz!")

playing = input("Do you want to play? ")

if playing.lower() != "yes": 
    quit()

print("Okay! Let's play :)")
score = 0

answer = input("What does CPU stand for? ").lower()
if answer == "central processing unit":
    print("Correct answer, Well Done!")
    score += 1
else:
    print("Incorrect answer!")

answer = input("What does GPU stand for? ").lower()
if answer == "graphics processing unit":
    print("Correct answer, Well Done!")
    score += 1
else:
    print("Incorrect answer!")

answer = input("What does RAM stand for? ").lower()
if answer == "random access memory":
    print("Correct answer, Well Done!")
    score += 1
else:
    print("Incorrect answer!")

answer = input("What does PSU stand for? ").lower()
if answer == "power supply":
    print("Correct answer, Well Done!")
    score += 1
else:
    print("Incorrect answer!")

print("you got " + str(score) + " question(s) correct!")
print("you got " + str((score/4) * 100) + " %.")