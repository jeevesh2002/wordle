import subprocess

def get_command(letters_of_word,in_word):
    letterrr=["","","","",""]
    not_start_regex = "[^"
    not_end_regex = "]"
    pipe = "|"
    for i in range(5):
        try:
            letterrr[i]=letters_of_word[i][1]
        except:
            for letter in letters_of_word[i][0]:
                letterrr[i]=letterrr[i]+letter
            if letterrr[i] != "":
                letterrr[i]=not_start_regex+letterrr[i]+not_end_regex
            else:
                letterrr[i]="."
    command = "".join(letterrr[i] for i in range(5))
    command = f"grep '{command}' five_letter.txt"
    command2=""
    for i in range(5):
        try:
            if temp := str(letters_of_word[i][1]):
                command2 += f" {pipe} grep '{temp}'"
        except:
            pass
        for l in in_word:
            command2 += f" {pipe} grep '{l}'"
    command = command + command2 + "| shuf -n1"
    return command

def update_list(colors,letters_of_word,in_word,word):
    for i in range(5):
        if colors[i] == "B":
            processed = "NO"
            for k in range(5):
                if (k != i and word[k]==word[i]) and colors[k] in ['Y', 'G']:
                    letters_of_word[i][0].append(word[i])
                    processed="YES"
                    break
            if processed=="NO":
                for j in range(5):
                    letters_of_word[j][0].append(word[i])
        elif colors[i] == "G":
            letters_of_word[i].append(word[i])
        elif colors[i] == "Y":
            letters_of_word[i][0].append(word[i])
            in_word.append(word[i])

letters_of_word = [[[],],[[],],[[],],[[],],[[],]]
in_word = []
wrong=[]
for _ in range(6):
    accepted="NO"
    word=""
    while accepted == "NO" :
        command = get_command(letters_of_word,in_word)
        word = subprocess.run(command, shell=True, capture_output=True, text=True)
        word = word.stdout
        print(command)      #debugging
        print(word[:-1])
        accepted = input("Enter YES if word is accepted else NO : ")
        if accepted == "YES":
            break
        subprocess.run(f"grep -v '{word[:-1]}' five_letter.txt > temp && cat temp > five_letter.txt && rm temp", shell=True)
    colors=input("Enter Colors: ").upper()
    if colors == "GGGGG":
        break
    update_list(colors,letters_of_word,in_word,word)















