# import util
#
# myHeap = util.PriorityQueue()
# myHeap.push([(3, 5), 'North'], 2)
# myHeap.push([[4,2], 'East'], 1)
# myHeap.push([(8, 7), 'South'], 5)
# myHeap.push([(1,1), 'North'], 4)
# myHeap.push([(10,6), 'West'], 7)
#
# state = [l[2][0] for l in myHeap.heap]
# print(state)
#
# path = [l[2] for l in myHeap.heap]
# print(path)
# print(path[1])
# x = (1,1)
# # if x in
# p = [y[1] for y in path if y[0] == x]
# print("Should be North",p)
corners = ((1,1), (1,'top'), ('right', 1), ('right', 'top'))
x = [False for i in corners]
print(x)
z = ((2,3), x)
y = z[1]

y[2] = True
print('y', y)

p = z[1][:]
p[2] = True

print('p', p)
print(z[0])
print(z[1])
print(z[1][:])
# def trying():
#     x = 2
#     z= 3
#     return x,z
#
#
# a = trying()
# print(type(a))
# print(a)
