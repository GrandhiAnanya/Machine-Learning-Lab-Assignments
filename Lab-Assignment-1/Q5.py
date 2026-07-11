import random
def mean(list):
   sum = 0
   for i in list:
     sum += i

   mean = sum / len(list)
   print("mean of the list: ",mean)

def median(list):
    if len(list)%2==0:
     pos1=(len(list)//2)-1
     pos2=(len(list)//2)
     median=(list[pos1]+list[pos2])/2
     print("\nmedian of the list: ",median)
    else:
      pos = (len(list)) // 2
      median = list[pos]
      print("\nmedian of the list: ",median)

def mode(list):
   frequency = {}

   for i in list:
     if i in frequency:
        frequency[i] += 1
     else:
        frequency[i] = 1

   max_freq = max(frequency.values())

   mode = []
   for key in frequency:
     if frequency[key] == max_freq:
        mode.append(key)
   print("\nmode of the list: ",mode)
   
   
  
list=[]
for i in range(100):
    list.append(random.randint(100, 150))


mean(list)
median(list)
mode(list)




