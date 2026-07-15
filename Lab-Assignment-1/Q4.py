def transpose(n,m):
    m1 = []
    print("\nEnter the elements of Matrix :")
    for i in range(n):
      row = []
      for j in range(m):
        e = int(input(f"Enter element [{i}][{j}]: "))
        row.append(e)
      m1.append(row)

    m2=[]
    for i in range(m):
      row=[]
      for j in range(n):
        row.append(0)
      m2.append(row)

    for i in range(n):
      for j in range(m):
        m2[j][i]=m1[i][j]

    return m1,m2



n = int(input("Enter the number of rows in Matrix 1: "))
m = int(input("Enter the number of columns in Matrix 1: "))
m1,m2=transpose(n,m)
print("original matix:\n")
for i in (m1):
      print(i)
print("\ntranspose of the matrix:\n")
for i in (m2):
      print(i)

