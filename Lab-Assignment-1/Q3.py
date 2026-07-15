def similarity(list1,list2):
    count=0

    for i in range(0,len(list1)):
      for j in range(0,len(list2)):
        if list1[i]==list2[j]:
            count+=1

    return count


list1 = [2, 5, 8, 10, 15, 20]
list2 = [1, 5, 7, 10, 14, 20]
count=similarity(list1,list2)
print("number of elements similar in the list are : ",count)
