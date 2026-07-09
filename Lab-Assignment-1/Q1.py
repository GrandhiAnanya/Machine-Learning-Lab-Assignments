s = input ("enter a string\n")
count_v = 0
count_c = 0
for i in s:
    if i.isalpha():
      if i in 'aeiouAEIOU':
        count_v+=1
      else:
        count_c+=1
print("number of vowels in the string are", count_v)
print("number of consonants in the string are", count_c)