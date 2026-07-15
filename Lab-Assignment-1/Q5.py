import random
def mean(lst):
   sum = 0
   for i in lst:
     sum += i

   mean = sum / len(lst)
   return mean

def median(lst):
    if len(lst)%2==0:
     pos1=(len(lst)//2)-1
     pos2=(len(lst)//2)
     median=(lst[pos1]+lst[pos2])/2
     return median
    else:
      pos = (len(lst)) // 2
      median = lst[pos]
      return median

def mode(lst):
   frequency = {}

   for i in lst:
     if i in frequency:
        frequency[i] += 1
     else:
        frequency[i] = 1

   max_freq = max(frequency.values())

   mode = []
   for key in frequency:
     if frequency[key] == max_freq:
        mode.append(key)
   return mode
   
   
  
lst=[]
for i in range(100):
    lst.append(random.randint(100, 150))

print (lst)
mean=mean(lst)
print("mean of the list: ",mean)
median=median(lst)
print("median of the list: ",median)
mode=mode(lst)
print("mode of the list: ",mode)




