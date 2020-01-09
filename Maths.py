first_half = []
point = 0.0001
first_half.append(point)
for i in range(49):
    point = point + 0.00041
    first_half.append(point)
print(first_half)
fnl=first_half[::-1] + first_half
print(sum(fnl))
print(len(fnl))

