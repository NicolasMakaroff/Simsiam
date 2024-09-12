import re
import matplotlib.pyplot as plt
import matplotlib as mpl

# Read the .out file
with open('output/simvae_emb.4904.out', 'r') as file:
    content = file.read()

    # Extract numbers after "Valid acc: "
    pattern = r"Valid acc: (\d+)"
    matches = re.findall(pattern, content)

# Extract individual values from matches
values = [float(match) for match in matches]
# Plot the values
x = range(0, len(values))


#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12



y = values
# Assuming x and y are defined
plt.plot(x, y)
plt.xlabel(r'Iteration')
plt.ylabel(r'Accuracy')
plt.legend()
plt.savefig('simvae_emb.png', dpi=300, bbox_inches='tight')
plt.close()

# Read the .out file
with open('output/simvae_proj.4903.out', 'r') as file:
    content = file.read()

    # Extract numbers after "Valid acc: "
    pattern = r"Valid acc: (\d+)"
    matches = re.findall(pattern, content)

# Extract individual values from matches
values = [float(match) for match in matches]
# Plot the values
x = range(0, len(values))


#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12



y = values
# Assuming x and y are defined
plt.plot(x, y)
plt.xlabel(r'Iteration')
plt.ylabel(r'Accuracy')
plt.legend()
plt.savefig('simvae_proj.png', dpi=300, bbox_inches='tight')
plt.close()

# Read the .out file
with open('output/simvae_mean.4905.out', 'r') as file:
    content = file.read()

    # Extract numbers after "Valid acc: "
    pattern = r"Valid acc: (\d+)"
    matches = re.findall(pattern, content)

# Extract individual values from matches
values = [float(match) for match in matches]
# Plot the values
x = range(0, len(values))


#mpl.rcParams['text.usetex'] = True
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['font.serif'] = 'Times'
mpl.rcParams['font.size'] = 12
mpl.rcParams['axes.labelsize'] = 12
mpl.rcParams['axes.titlesize'] = 12
mpl.rcParams['xtick.labelsize'] = 12
mpl.rcParams['ytick.labelsize'] = 12
mpl.rcParams['legend.fontsize'] = 12



y = values
# Assuming x and y are defined
plt.plot(x, y)
plt.xlabel(r'Iteration')
plt.ylabel(r'Accuracy')
plt.legend()
plt.savefig('simvae_mean.png', dpi=300, bbox_inches='tight')
plt.close()