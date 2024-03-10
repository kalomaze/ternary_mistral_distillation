import re
import matplotlib.pyplot as plt

# Read the contents of the file
with open('graff.txt', 'r') as file:
    contents = file.read()

# Use regular expressions to extract the batch numbers and loss values
batches = []
losses = []
pattern = r'Epoch:\s*\d+,\s*Batch:\s*(\d+),\s*Loss:\s*([\d.]+)'
matches = re.findall(pattern, contents)

for match in matches:
    batch, loss = match
    batches.append(int(batch))
    losses.append(float(loss))

# Create the matplotlib graph
plt.figure(figsize=(10, 6))
plt.plot(batches, losses)
plt.xlabel('Batch')
plt.ylabel('Loss')
plt.title('Loss vs. Batch Number')
plt.show()
