import sys

# Read command line arguments and convert to a list of integers
arr = sys.argv[1].split(',')
my_numbers = [None]*len(arr)
for idx, arr_val in enumerate(arr):
    my_numbers[idx] = int(arr_val)

# Print
print(f'Before sorting {my_numbers}')

# My sorting (e.g. bubble sort)
# ADD HERE YOUR CODE

def bubbleSort(array):
    
  for i in range(len(array)):

    for j in range(0, len(array) - i - 1):

      if array[j] > array[j + 1]:

        temp = array[j]
        array[j] = array[j+1]
        array[j+1] = temp
        
bubbleSort(my_numbers)
        
# Print
print(f'After sorting {my_numbers}')
