def matrixmul(n,m,p,q):
    if m != p:
      print("Error: Matrices are not multipliable.")
      exit()
    

    else:
     m1 = []
     print("\nEnter the elements of Matrix 1:")
    for i in range(n):
      row = []
      for j in range(m):
        e = int(input(f"Enter element [{i}][{j}]: "))
        row.append(e)
      m1.append(row)


    m2 = []
    print("\nEnter the elements of Matrix 2:")
    for i in range(p):
      row = []
      for j in range(q):
        e = int(input(f"Enter element [{i}][{j}]: "))
        row.append(e)
      m2.append(row)


    mul = []
    for i in range(n):
     row = []
     for j in range(q):
        row.append(0)
     mul.append(row)


    for i in range(n):
     for j in range(q):
        for k in range(m):
            mul[i][j] += m1[i][k] * m2[k][j]
  
    return mul
    
   

n = int(input("Enter the number of rows in Matrix 1: "))
m = int(input("Enter the number of columns in Matrix 1: "))

p = int(input("Enter the number of rows in Matrix 2: "))
q = int(input("Enter the number of columns in Matrix 2: "))

mul=matrixmul(n,m,p,q)

print("\nProduct of the two matrices:")
for i in range(n):
     print(mul[i])

